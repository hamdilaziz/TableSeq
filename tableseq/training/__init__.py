"""Training utilities for TableSeq."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FastTableMetrics",
    "TableSeqTrainer",
    "TableSeqTrainingConfig",
    "TokenSubstitutionNoise",
    "clean_tableseq_html",
    "compute_box_metrics_from_strings",
    "extract_quantized_boxes",
]


def __getattr__(name: str) -> Any:
    if name in {"TableSeqTrainer", "TableSeqTrainingConfig"}:
        from .trainer import TableSeqTrainer, TableSeqTrainingConfig

        return {"TableSeqTrainer": TableSeqTrainer, "TableSeqTrainingConfig": TableSeqTrainingConfig}[name]
    if name == "TokenSubstitutionNoise":
        from .noise import TokenSubstitutionNoise

        return TokenSubstitutionNoise
    if name == "FastTableMetrics":
        from .fast_metrics import FastTableMetrics

        return FastTableMetrics
    if name == "clean_tableseq_html":
        from .metrics import clean_tableseq_html

        return clean_tableseq_html
    if name in {"compute_box_metrics_from_strings", "extract_quantized_boxes"}:
        from .box_metrics import compute_box_metrics_from_strings, extract_quantized_boxes

        return {
            "compute_box_metrics_from_strings": compute_box_metrics_from_strings,
            "extract_quantized_boxes": extract_quantized_boxes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
