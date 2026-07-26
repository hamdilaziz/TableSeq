import pickle
from types import SimpleNamespace

from PIL import Image

from tableseq.data import TableSeqDataset, TableSeqDatasetConfig


class _Tokenizer:
    bos_token_id = 0
    pad_token_id = 1

    def __init__(self) -> None:
        coordinate_tokens = {f"<x_{i}>": i + 10 for i in range(101)}
        coordinate_tokens.update({f"<y_{i}>": i + 200 for i in range(101)})
        self._vocab = {"<s>": 0, "<pad>": 1, "<html>": 2, **coordinate_tokens}

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, add_special_tokens, return_attention_mask):
        return SimpleNamespace(input_ids=[0, 3] if add_special_tokens else [2])


def test_dataset_scales_original_coordinate_labels_with_image(tmp_path) -> None:
    (tmp_path / "train").mkdir()
    Image.new("RGB", (100, 50), "white").save(tmp_path / "train" / "sample.png")
    labels = {
        "ground_truth": {
            "train": {
                "sample.png": {
                    "text": "<td><x_10><y_5>A<x_20><y_10></td>",
                    "prompt": "<html>",
                }
            }
        }
    }
    with open(tmp_path / "labels.pkl", "wb") as handle:
        pickle.dump(labels, handle)

    dataset = TableSeqDataset(
        TableSeqDatasetConfig(
            dataset_root=str(tmp_path),
            label_file="labels.pkl",
            resize_factor=1.5,
            normalize=False,
        ),
        split="train",
        tokenizer=_Tokenizer(),
    )

    sample = dataset[0]
    assert sample["img"].shape[:2] == (75, 150)
    assert sample["raw_label"] == "<td><x_15><y_8>A<x_30><y_15></td>"
    assert sample["original_raw_label"] == "<td><x_10><y_5>A<x_20><y_10></td>"
    assert sample["image_geometry"]["scale_x"] == 1.5
    assert sample["image_geometry"]["scale_y"] == 1.5


def test_dataset_can_keep_pre_scaled_coordinate_labels(tmp_path) -> None:
    (tmp_path / "train").mkdir()
    Image.new("RGB", (100, 50), "white").save(tmp_path / "train" / "sample.png")
    labels = {
        "ground_truth": {
            "train": {
                "sample.png": {
                    "text": "<x_15><y_8><x_30><y_15>",
                    "prompt": "<html>",
                }
            }
        }
    }
    with open(tmp_path / "labels.pkl", "wb") as handle:
        pickle.dump(labels, handle)

    dataset = TableSeqDataset(
        TableSeqDatasetConfig(
            dataset_root=str(tmp_path),
            label_file="labels.pkl",
            resize_factor=1.5,
            coordinate_labels_in_original_image_space=False,
            normalize=False,
        ),
        split="train",
        tokenizer=_Tokenizer(),
    )

    assert dataset[0]["raw_label"] == "<x_15><y_8><x_30><y_15>"
