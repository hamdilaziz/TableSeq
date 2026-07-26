from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tableseq import TableSeqModel
from tableseq.coordinates import resolve_resize_factor
from tableseq.data import TableSeqCollator, TableSeqDataset, TableSeqDatasetConfig
from tableseq.modeling import extract_sequences
from tableseq.models.encoder import TableSeqEncoderConfig
from tableseq.training.box_metrics import (
    average_metric_dicts,
    build_coordinate_vocab_masks,
    compute_box_metrics_from_strings,
    coordinate_token_accuracy_from_logits,
)
from tableseq.training.fast_metrics import FastTableMetrics
from tableseq.training.metrics import clean_tableseq_html
from tableseq.training.trainer import build_teacher_forcing_inputs


def sequence_length(seq: torch.Tensor, pad_id: Optional[int], eos_id: Optional[int]) -> int:
    values = seq.detach().cpu().tolist()

    if eos_id is not None and eos_id in values:
        return values.index(eos_id) + 1

    if pad_id is not None and pad_id in values:
        return values.index(pad_id)

    return len(values)


def build_split_loader(
    data_config: TableSeqDatasetConfig,
    tokenizer,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TableSeqDataset(
        config=data_config,
        split=split,
        tokenizer=tokenizer,
        max_samples=None,
    )

    collator = TableSeqCollator(
        tokenizer=tokenizer,
        image_padding_value=data_config.image_padding_value,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def _resolve_label_coordinate_space(args: argparse.Namespace, checkpoint_metadata: dict[str, Any]) -> tuple[bool, str]:
    if args.labels_in_original_space is not None:
        return bool(args.labels_in_original_space), "explicit"

    data_config = checkpoint_metadata.get("data_config")
    if isinstance(data_config, dict) and data_config.get("coordinate_labels_in_original_image_space") is not None:
        return bool(data_config["coordinate_labels_in_original_image_space"]), "checkpoint"

    return True, "fallback"


@torch.no_grad()
def compute_teacher_forced_accuracy_rows(
    model: TableSeqModel,
    batch: dict[str, Any],
    device: torch.device,
    max_length: int,
    x_vocab_mask: torch.Tensor,
    y_vocab_mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Compute teacher-forced token accuracies without calculating a loss."""
    tokenizer = model.decoder.tokenizer
    pad_token_id = int(tokenizer.pad_token_id)

    images = batch["imgs"].to(device, non_blocking=True)
    image_attention_mask = batch.get("image_attention_mask")
    if image_attention_mask is not None:
        image_attention_mask = image_attention_mask.to(device, non_blocking=True)

    labels = batch["labels"].to(device, non_blocking=True)
    labels_len = batch["labels_len"]
    if torch.is_tensor(labels_len):
        labels_len = labels_len.detach().cpu().tolist()
    labels_len = [int(value) for value in labels_len]

    token_prompts = [
        prompt.to(device, non_blocking=True)
        if torch.is_tensor(prompt)
        else torch.tensor(prompt, dtype=torch.long, device=device)
        for prompt in batch["token_prompt"]
    ]

    input_ids, target_labels = build_teacher_forcing_inputs(
        labels=labels,
        labels_len=labels_len,
        token_prompts=token_prompts,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )
    decoder_attention_mask = input_ids.ne(pad_token_id).long()

    decoder_output = model(
        images=images,
        input_ids=input_ids,
        labels=None,
        attention_mask=decoder_attention_mask,
        image_attention_mask=image_attention_mask,
        use_structure_bias=False,
        use_cache=False,
    )
    logits = decoder_output["logits"] if isinstance(decoder_output, dict) else decoder_output.logits
    predictions = torch.argmax(logits.detach(), dim=-1)

    rows: list[dict[str, float]] = []
    for sample_logits, sample_pred, sample_labels in zip(logits, predictions, target_labels):
        valid = sample_labels.ne(pad_token_id)
        token_count = int(valid.sum().item())
        token_correct = int(sample_pred.eq(sample_labels).logical_and(valid).sum().item())
        row: dict[str, float] = {
            "token_count": float(token_count),
            "token_correct": float(token_correct),
            "token_acc": float(token_correct / token_count) if token_count else 0.0,
        }
        row.update(
            coordinate_token_accuracy_from_logits(
                logits=sample_logits.unsqueeze(0),
                labels=sample_labels.unsqueeze(0),
                x_vocab_mask=x_vocab_mask,
                y_vocab_mask=y_vocab_mask,
            )
        )
        for axis in ("coord", "x", "y"):
            count_key = f"{axis}_token_count"
            acc_key = f"{axis}_token_acc"
            if count_key in row and acc_key in row:
                row[f"{axis}_token_correct"] = float(row[count_key] * row[acc_key])
        rows.append(row)
    return rows


def _weighted_accuracy(rows: list[dict[str, Any]], prefix: str) -> tuple[float | None, float]:
    count_key = f"{prefix}_count"
    correct_key = f"{prefix}_correct"
    total = float(sum(float(row.get(count_key, 0.0)) for row in rows))
    if total <= 0:
        return None, 0.0
    correct = float(sum(float(row.get(correct_key, 0.0)) for row in rows))
    return correct / total, total


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    model_config = {
        "encoder": TableSeqEncoderConfig(),
        "decoder": {
            "tokenizer_path": args.tokenizer_path,
            "max_length": args.max_length,
        },
        "checkpoints_path": args.checkpoint,
        "strict_load": not args.non_strict_load,
        "device": str(device),
    }

    model = TableSeqModel(model_config).to(device)
    model.eval()

    resize_factor, resize_source = resolve_resize_factor(
        requested=args.resize_factor,
        checkpoint_metadata=model.checkpoint_metadata,
        fallback=1.0,
    )
    if resize_source == "fallback":
        print(
            "Checkpoint does not contain a saved resize factor; using 1.0. "
            "Pass --resize-factor explicitly for older checkpoints trained at another scale."
        )
    elif resize_source == "explicit_override":
        print("Explicit resize factor overrides the value stored in the checkpoint.")

    labels_in_original_space, labels_space_source = _resolve_label_coordinate_space(
        args,
        model.checkpoint_metadata,
    )

    tokenizer = model.decoder.tokenizer

    data_config = TableSeqDatasetConfig(
        dataset_root=args.dataset_root,
        label_file=args.label_file,
        resize_factor=resize_factor,
        coordinate_labels_in_original_image_space=labels_in_original_space,
        normalize=True,
        image_padding_value=0.0,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    loader = build_split_loader(
        data_config=data_config,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{args.split}_predictions.jsonl"
    csv_path = output_dir / f"{args.split}_metrics.csv"
    summary_path = output_dir / f"{args.split}_summary.json"

    metrics = FastTableMetrics()
    x_vocab_mask, y_vocab_mask = build_coordinate_vocab_masks(tokenizer)

    rows: list[dict[str, Any]] = []

    struct_seq_scores: list[float] = []
    steds_scores: list[float] = []
    teds_scores: list[float] = []
    box_metric_dicts: list[dict[str, float]] = []
    gen_lengths: list[int] = []
    processed_batches = 0

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        pbar = tqdm(loader, desc=f"Evaluating {args.split}", total=len(loader))

        for batch in pbar:
            images = batch["imgs"].to(device, non_blocking=True)

            image_attention_mask = batch.get("image_attention_mask")
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask.to(device, non_blocking=True)

            batch_size = images.size(0)

            input_ids, _ = model.build_input_ids_from_prompt(
                "<html>",
                batch_size=batch_size,
                device=device,
            )

            output = model.generate(
                images=images,
                input_ids=input_ids,
                image_attention_mask=image_attention_mask,
                max_length=args.max_length,
                use_cache=True,
                use_structure_bias=False,
                return_dict_in_generate=True,
            )
            sequences = extract_sequences(output)

            predictions = [tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
            teacher_forced_rows = [dict() for _ in range(batch_size)]
            if args.compute_token_accuracy:
                teacher_forced_rows = compute_teacher_forced_accuracy_rows(
                    model=model,
                    batch=batch,
                    device=device,
                    max_length=args.max_length,
                    use_structure_bias=False,
                                x_vocab_mask=x_vocab_mask,
                    y_vocab_mask=y_vocab_mask,
                )

            names = batch.get("names", [f"sample_{len(rows) + i}" for i in range(batch_size)])
            paths = batch.get("paths", [None for _ in range(batch_size)])
            references = batch.get("raw_labels", [None for _ in range(batch_size)])
            original_references = batch.get("original_raw_labels", references)
            geometries = batch.get("image_geometries", [None for _ in range(batch_size)])

            for name, path, pred_raw, gt_raw, gt_original, geometry, seq, teacher_forced in zip(
                names,
                paths,
                predictions,
                references,
                original_references,
                geometries,
                sequences,
                teacher_forced_rows,
            ):
                pred_clean = clean_tableseq_html(pred_raw)
                gt_clean = clean_tableseq_html(gt_raw) if gt_raw is not None else None
                gen_len = sequence_length(seq, pad_id=pad_id, eos_id=eos_id)
                gen_lengths.append(gen_len)

                row: dict[str, Any] = {
                    "name": name,
                    "path": path,
                    "prediction_raw": pred_raw,
                    "prediction_clean": pred_clean,
                    "reference_raw": gt_raw,
                    "reference_original_raw": gt_original,
                    "reference_clean": gt_clean,
                    "image_geometry": geometry,
                    "gen_len": gen_len,
                    **teacher_forced,
                }

                if gt_raw is not None:
                    if args.compute_struct_seq:
                        try:
                            score = metrics.score_structure_sequence(gt_raw=gt_raw, pred_raw=pred_raw, key=name)
                            row["struct_seq"] = float(score)
                            struct_seq_scores.append(float(score))
                        except Exception as exc:
                            row["struct_seq_error"] = str(exc)

                    if args.compute_steds:
                        try:
                            score = metrics.score_steds(gt_raw=gt_raw, pred_raw=pred_raw, key=name)
                            row["steds"] = float(score)
                            steds_scores.append(float(score))
                        except Exception as exc:
                            row["steds_error"] = str(exc)

                    if args.compute_teds:
                        try:
                            score = metrics.score_teds(gt_raw=gt_raw, pred_raw=pred_raw, key=name)
                            row["teds"] = float(score)
                            teds_scores.append(float(score))
                        except Exception as exc:
                            row["teds_error"] = str(exc)

                    if args.compute_box_metrics:
                        try:
                            scale_x = float(geometry.get("scale_x", 1.0)) if isinstance(geometry, dict) else 1.0
                            scale_y = (
                                float(geometry.get("scale_y", scale_x))
                                if isinstance(geometry, dict)
                                else scale_x
                            )
                            box_values = compute_box_metrics_from_strings(
                                reference=gt_raw,
                                prediction=pred_raw,
                                image_scale_x=scale_x,
                                image_scale_y=scale_y,
                            ).to_dict(prefix="")
                            row.update(box_values)
                            box_metric_dicts.append(box_values)
                        except Exception as exc:
                            row["box_metrics_error"] = str(exc)

                rows.append(row)
                f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

            processed_batches += 1
            postfix: dict[str, Any] = {
                "samples": len(rows),
                "len": f"{float(np.mean(gen_lengths)):.1f}" if gen_lengths else "0",
            }
            if struct_seq_scores:
                postfix["struct_seq"] = f"{float(np.mean(struct_seq_scores)):.4f}"
            if steds_scores:
                postfix["steds"] = f"{float(np.mean(steds_scores)):.4f}"
            if teds_scores:
                postfix["teds"] = f"{float(np.mean(teds_scores)):.4f}"
            if box_metric_dicts:
                running_boxes = average_metric_dicts(box_metric_dicts)
                if "box_iou" in running_boxes:
                    postfix["box_iou"] = f"{running_boxes['box_iou']:.4f}"
                if "box_f1_50" in running_boxes:
                    postfix["box_f1_50"] = f"{running_boxes['box_f1_50']:.4f}"
            if args.compute_token_accuracy:
                token_acc, _ = _weighted_accuracy(rows, "token")
                if token_acc is not None:
                    postfix["token_acc"] = f"{token_acc:.4f}"
            pbar.set_postfix(**postfix)

    box_summary = average_metric_dicts(box_metric_dicts)
    validation_metrics: dict[str, float] = {}
    if struct_seq_scores:
        validation_metrics["valid_struct_seq"] = float(np.mean(struct_seq_scores))
    if steds_scores:
        validation_metrics["valid_steds"] = float(np.mean(steds_scores))
    if teds_scores:
        validation_metrics["valid_teds"] = float(np.mean(teds_scores))
    for key, value in box_summary.items():
        validation_metrics[f"valid_{key}"] = float(value)
    if gen_lengths:
        validation_metrics["valid_gen_len"] = float(np.mean(gen_lengths))
    validation_metrics["valid_generation_batches"] = float(processed_batches)
    validation_metrics["valid_generation_samples"] = float(len(rows))

    if args.compute_token_accuracy:
        for prefix, metric_name, count_name in (
            ("token", "valid_token_acc", "valid_token_count"),
            ("coord_token", "valid_coord_token_acc", "valid_coord_token_count"),
            ("x_token", "valid_x_token_acc", "valid_x_token_count"),
            ("y_token", "valid_y_token_acc", "valid_y_token_count"),
        ):
            accuracy, count = _weighted_accuracy(rows, prefix)
            if accuracy is not None:
                validation_metrics[metric_name] = float(accuracy)
            validation_metrics[count_name] = float(count)

    metric_fields = [
        "name",
        "path",
        "gen_len",
        "token_acc",
        "coord_token_acc",
        "x_token_acc",
        "y_token_acc",
        "struct_seq",
        "steds",
        "teds",
        "n_gt_boxes",
        "n_pred_boxes",
        "box_count_abs_diff",
        "box_iou",
        "box_iou_matched",
        "box_f1_50",
        "box_f1_75",
        "box_l1",
        "prediction_clean",
        "reference_clean",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "label_file": args.label_file,
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "resize_factor": resize_factor,
        "resize_factor_source": resize_source,
        "coordinate_labels_in_original_image_space": labels_in_original_space,
        "coordinate_label_space_source": labels_space_source,
        "computed_metrics": {
            "struct_seq": args.compute_struct_seq,
            "steds": args.compute_steds,
            "teds": args.compute_teds,
            "box_metrics": args.compute_box_metrics,
            "teacher_forced_token_accuracy": args.compute_token_accuracy,
        },
        **validation_metrics,
    }

    if args.compute_box_metrics and box_metric_dicts:
        total_gt_boxes = sum(int(values.get("n_gt_boxes", 0)) for values in box_metric_dicts)
        if total_gt_boxes == 0:
            summary["box_metrics_warning"] = (
                "No reference coordinate boxes were found. Box IoU/F1 cannot validate localization; "
                "check the label file and coordinate-label-space setting."
            )
            print(f"WARNING: {summary['box_metrics_warning']}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved predictions: {jsonl_path}")
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TableSeq with the same generation metrics used during training."
    )

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-file", default="labels_with_boxes_org.pkl")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", default="outputs/eval")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=None,
        help=(
            "Image resize factor used by the checkpoint. New checkpoints store it automatically; "
            "pass it explicitly for older checkpoints."
        ),
    )

    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--labels-in-original-space",
        dest="labels_in_original_space",
        action="store_true",
        help="Coordinate tokens in the label file refer to original-image pixels (default for new datasets).",
    )
    label_group.add_argument(
        "--labels-already-resized",
        dest="labels_in_original_space",
        action="store_false",
        help="Coordinate tokens in the label file already refer to the resized model-input images.",
    )
    parser.set_defaults(labels_in_original_space=None)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--non-strict-load", action="store_true")

    parser.add_argument(
        "--compute-struct-seq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute linearized structure sequence similarity.",
    )
    parser.add_argument(
        "--compute-steds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute structure-only TEDS.",
    )
    parser.add_argument(
        "--compute-teds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute full content-aware TEDS (slower).",
    )
    parser.add_argument(
        "--compute-box-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the same generated box metrics used during training validation.",
    )
    parser.add_argument(
        "--compute-token-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute teacher-forced token and coordinate-token accuracies without computing loss.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
