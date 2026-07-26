from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor


class TokenSubstitutionNoise:
    """Token-level teacher-forcing noise from a JSON substitution file.

    The expected JSON format is:

        {
            "<td>": ["<th>", "</td>"],
            "<tr>": ["</tr>"]
        }

    Keys and values may also be integer token ids represented as JSON strings.
    Noise is applied only to decoder inputs. The target labels remain clean.
    """

    def __init__(
        self,
        substitution_dict: dict[int, list[int]],
        pad_token_id: int,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
    ) -> None:
        self.substitution_dict = substitution_dict
        self.pad_token_id = int(pad_token_id)
        self.protected_ids = {int(pad_token_id)}
        for token_id in [bos_token_id, eos_token_id, unk_token_id]:
            if token_id is not None:
                self.protected_ids.add(int(token_id))
        self.last_stats = {"eligible": 0, "changed": 0}

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Substitution file not found: {path}")
        if path.suffix.lower() != ".json":
            raise ValueError(f"Token substitution file must be a JSON file, got: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_token_id(value: Any, tokenizer) -> Optional[int]:
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            token_id = tokenizer.convert_tokens_to_ids(value)
            if token_id is None:
                return None
            if token_id == tokenizer.unk_token_id and value != tokenizer.unk_token:
                return None
            return int(token_id)
        return None

    @classmethod
    def from_json(cls, path: str | Path, tokenizer) -> "TokenSubstitutionNoise":
        raw = cls._read_json(path)
        substitution_dict: dict[int, list[int]] = {}
        skipped_sources = 0
        skipped_replacements = 0

        for src, replacements in raw.items():
            src_id = cls._to_token_id(src, tokenizer)
            if src_id is None:
                skipped_sources += 1
                continue
            if not isinstance(replacements, (list, tuple)):
                replacements = [replacements]

            replacement_ids: list[int] = []
            for repl in replacements:
                repl_id = cls._to_token_id(repl, tokenizer)
                if repl_id is None:
                    skipped_replacements += 1
                    continue
                replacement_ids.append(int(repl_id))

            if replacement_ids:
                substitution_dict[int(src_id)] = replacement_ids

        obj = cls(
            substitution_dict=substitution_dict,
            pad_token_id=int(tokenizer.pad_token_id),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            unk_token_id=getattr(tokenizer, "unk_token_id", None),
        )
        obj.skipped_sources = skipped_sources
        obj.skipped_replacements = skipped_replacements
        return obj

    def __call__(
        self,
        labels: Tensor,
        labels_len: list[int],
        token_prompts: list[Tensor],
        error_rate: float,
    ) -> Tensor:
        if error_rate <= 0.0:
            self.last_stats = {"eligible": 0, "changed": 0}
            return labels

        noisy = labels.clone()
        eligible = 0
        changed = 0

        for b in range(labels.size(0)):
            prompt_len = int(token_prompts[b].numel())
            start = prompt_len + 1  # skip <s> and prompt
            end = max(start, int(labels_len[b]) - 1)  # do not corrupt final </s>

            for t in range(start, end):
                current_id = int(labels[b, t].item())
                if current_id in self.protected_ids:
                    continue
                replacements = self.substitution_dict.get(current_id)
                if not replacements:
                    continue

                eligible += 1
                if torch.rand((), device=labels.device).item() < error_rate:
                    idx = torch.randint(low=0, high=len(replacements), size=(1,), device=labels.device).item()
                    noisy[b, t] = int(replacements[idx])
                    changed += 1

        self.last_stats = {"eligible": eligible, "changed": changed}
        return noisy
