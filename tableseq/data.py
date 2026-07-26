from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .coordinates import COORD_TOKEN_RE, ImageGeometry, scale_coordinate_tokens, validate_resize_factor


@dataclass
class TableSeqDatasetConfig:
    dataset_root: str
    label_file: str = "labels_with_boxes_org.pkl"

    mean: tuple[float, float, float] = (
        238.20993215,
        238.2207874,
        238.31686326,
    )
    std: tuple[float, float, float] = (
        47.43008097,
        47.65201466,
        47.5893778,
    )

    resize_factor: float = 1.0
    coordinate_labels_in_original_image_space: bool = True
    strict_coordinate_vocabulary: bool = True
    normalize: bool = True
    image_padding_value: float = 0.0
    default_prompt: str = "<html>"

    max_height: Optional[int] = None
    max_width: Optional[int] = None

    train_batch_size: int = 2
    valid_batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    train_shuffle: bool = True
    seed: int = 42

    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    random_train_subset: bool = False
    random_valid_subset: bool = False
    skip_missing_images: bool = True


def load_image_for_model(
    image_path: str | Path,
    config: TableSeqDatasetConfig,
) -> tuple[np.ndarray, ImageGeometry]:
    """Load and preprocess one image while preserving its exact geometry."""
    img = Image.open(image_path).convert("RGB")
    original_width, original_height = img.size
    resize_factor = validate_resize_factor(config.resize_factor)

    if resize_factor != 1.0:
        resized_width = max(1, int(original_width * resize_factor))
        resized_height = max(1, int(original_height * resize_factor))
        img = img.resize((resized_width, resized_height), resample=Image.BICUBIC)

    if config.max_height is not None or config.max_width is not None:
        current_width, current_height = img.size
        max_h = config.max_height or current_height
        max_w = config.max_width or current_width
        ratio = max(current_height / max_h, current_width / max_w)
        if ratio > 1:
            new_h = max(1, int(np.ceil(current_height / ratio)))
            new_w = max(1, int(np.ceil(current_width / ratio)))
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    model_width, model_height = img.size
    geometry = ImageGeometry(
        original_width=original_width,
        original_height=original_height,
        model_width=model_width,
        model_height=model_height,
    )

    img_arr = np.array(img).astype(np.float32)
    if config.normalize:
        mean = np.array(config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(config.std, dtype=np.float32).reshape(1, 1, 3)
        img_arr = (img_arr - mean) / std

    return img_arr.astype(np.float32), geometry


def pad_sequences_1d(sequences: list[list[int]], pad_token_id: int) -> torch.LongTensor:
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), fill_value=pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def pad_images_bottom_right(
    images: list[np.ndarray],
    padding_value: float,
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    """Pad images and return the matching pixel mask.

    Input images are H x W x C arrays. The output image tensor is B x C x H x W.
    The mask is B x H x W, with 1 for real image pixels and 0 for padding.
    """
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    channels = images[0].shape[2]

    batch = np.full((len(images), max_h, max_w, channels), fill_value=padding_value, dtype=np.float32)
    mask = np.zeros((len(images), max_h, max_w), dtype=np.int64)

    for i, img in enumerate(images):
        h, w, _ = img.shape
        batch[i, :h, :w, :] = img
        mask[i, :h, :w] = 1

    imgs = torch.tensor(batch, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    image_attention_mask = torch.tensor(mask, dtype=torch.long)
    return imgs, image_attention_mask


class TableSeqDataset(Dataset):
    """Dataset for TableSeq training.

    Expected labels file:

        {
            "ground_truth": {
                "train": {"image.png": {"text": "<table>...</table>", "prompt": "<html>"}},
                "valid": {},
                "test": {}
            }
        }
    """

    def __init__(
        self,
        config: TableSeqDatasetConfig,
        split: str,
        tokenizer,
        max_samples: Optional[int] = None,
        random_subset: bool = False,
    ) -> None:
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_root = Path(config.dataset_root)
        self.label_path = self.dataset_root / config.label_file
        self.tokenizer_vocab = set(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else set()
        self.samples = self._load_samples(max_samples=max_samples, random_subset=random_subset)

    def _load_samples(self, max_samples: Optional[int], random_subset: bool) -> list[dict[str, Any]]:
        with open(self.label_path, "rb") as f:
            info = pickle.load(f)

        if "ground_truth" not in info:
            raise KeyError(f"{self.label_path} does not contain a 'ground_truth' key.")
        if self.split not in info["ground_truth"]:
            raise KeyError(
                f"{self.label_path} does not contain split '{self.split}'. "
                f"Available splits: {list(info['ground_truth'].keys())}"
            )

        gt = info["ground_truth"][self.split]
        image_names = list(gt.keys())
        if max_samples is not None:
            if random_subset:
                split_offset = {"train": 0, "valid": 1, "test": 2}.get(self.split, 3)
                rng = random.Random(self.config.seed + split_offset)
                rng.shuffle(image_names)
            image_names = image_names[:max_samples]

        samples: list[dict[str, Any]] = []
        skipped = 0
        for image_name in image_names:
            entry = gt[image_name]
            if isinstance(entry, dict):
                text = entry["text"]
                prompt = entry.get("prompt", self.config.default_prompt)
                metadata = {k: v for k, v in entry.items() if k not in {"text", "prompt"}}
            else:
                text = entry
                prompt = self.config.default_prompt
                metadata = {}

            image_path = self.dataset_root / self.split / image_name
            if not image_path.exists():
                image_path = self.dataset_root / image_name
            if not image_path.exists():
                if self.config.skip_missing_images:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Image not found: {self.dataset_root / self.split / image_name}")

            samples.append(
                {
                    "name": image_name,
                    "path": str(image_path),
                    "text": text,
                    "prompt": prompt,
                    "metadata": metadata,
                }
            )

        if skipped:
            print(f"[{self.split}] skipped {skipped} samples with missing images.")
        print(f"[{self.split}] loaded {len(samples)} samples from {self.label_path}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> tuple[np.ndarray, ImageGeometry]:
        return load_image_for_model(image_path, self.config)

    def _validate_coordinate_vocabulary(self, text: str, image_name: str) -> None:
        if not self.config.strict_coordinate_vocabulary or not self.tokenizer_vocab:
            return
        missing = sorted(
            {
                match.group(0)
                for match in COORD_TOKEN_RE.finditer(text)
                if match.group(0) not in self.tokenizer_vocab
            }
        )
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8} more)"
            raise ValueError(
                f"Coordinate tokens are missing from the tokenizer for {image_name}: {preview}{suffix}. "
                "The tokenizer coordinate range must cover the selected resize factor."
            )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=True, return_attention_mask=False).input_ids

    def _encode_prompt(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt, add_special_tokens=False, return_attention_mask=False).input_ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img, geometry = self._load_image(sample["path"])

        original_text = sample["text"]
        model_text = original_text
        if self.config.coordinate_labels_in_original_image_space:
            model_text = scale_coordinate_tokens(
                original_text,
                scale_x=geometry.scale_x,
                scale_y=geometry.scale_y,
            )

        self._validate_coordinate_vocabulary(model_text, sample["name"])
        token_label = self._encode_text(model_text)
        token_prompt = self._encode_prompt(sample["prompt"])
        metadata = dict(sample["metadata"])
        metadata["image_geometry"] = geometry.to_dict()

        return {
            "name": sample["name"],
            "path": sample["path"],
            "img": img,
            "raw_label": model_text,
            "original_raw_label": original_text,
            "prompt": sample["prompt"],
            "token_label": token_label,
            "token_prompt": token_prompt,
            "metadata": metadata,
            "image_geometry": geometry.to_dict(),
        }


class TableSeqCollator:
    """Collate function compatible with TableSeqTrainer.

    It builds labels as ``<s> <html> target </s>``.
    """

    def __init__(self, tokenizer, image_padding_value: float = 0.0) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.image_padding_value = float(image_padding_value)

    def _insert_prompt_after_bos(self, token_label: list[int], token_prompt: list[int]) -> list[int]:
        if len(token_label) == 0:
            raise ValueError("Empty token_label.")
        if self.bos_token_id is not None and token_label[0] == self.bos_token_id:
            return token_label[:1] + token_prompt + token_label[1:]
        return token_prompt + token_label

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        names = [sample["name"] for sample in batch]
        paths = [sample["path"] for sample in batch]
        raw_labels = [sample["raw_label"] for sample in batch]
        original_raw_labels = [sample.get("original_raw_label", sample["raw_label"]) for sample in batch]
        image_geometries = [sample.get("image_geometry") for sample in batch]
        images = [sample["img"] for sample in batch]

        full_labels: list[list[int]] = []
        labels_len: list[int] = []
        token_prompts: list[torch.Tensor] = []

        for sample in batch:
            token_prompt = list(sample["token_prompt"])
            full_label = self._insert_prompt_after_bos(list(sample["token_label"]), token_prompt)
            full_labels.append(full_label)
            labels_len.append(len(full_label))
            token_prompts.append(torch.tensor(token_prompt, dtype=torch.long))

        labels = pad_sequences_1d(full_labels, pad_token_id=self.pad_token_id)
        imgs, image_attention_mask = pad_images_bottom_right(images, padding_value=self.image_padding_value)

        return {
            "names": names,
            "paths": paths,
            "imgs": imgs,
            "image_attention_mask": image_attention_mask,
            "labels": labels,
            "labels_len": labels_len,
            "token_prompt": token_prompts,
            "raw_labels": raw_labels,
            "original_raw_labels": original_raw_labels,
            "image_geometries": image_geometries,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_tableseq_dataloaders(config: TableSeqDatasetConfig, tokenizer) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_dataset = TableSeqDataset(
        config=config,
        split="train",
        tokenizer=tokenizer,
        max_samples=config.max_train_samples,
        random_subset=config.random_train_subset,
    )
    valid_dataset = TableSeqDataset(
        config=config,
        split="valid",
        tokenizer=tokenizer,
        max_samples=config.max_valid_samples,
        random_subset=config.random_valid_subset,
    )
    collator = TableSeqCollator(tokenizer=tokenizer, image_padding_value=config.image_padding_value)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, valid_loader
