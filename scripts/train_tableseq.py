from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tableseq import TableSeqModel
from tableseq.data import TableSeqDatasetConfig, build_tableseq_dataloaders
from tableseq.models.encoder import TableSeqEncoderConfig
from tableseq.training.trainer import TableSeqTrainer, TableSeqTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TableSeq.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-file", default="labels_with_boxes_org.pkl")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/tableseq_finetune")
    parser.add_argument("--token-substitution-json", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--valid-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument(
        "--labels-already-resized",
        action="store_true",
        help="Use this only when coordinate tokens in the label file already match the resized model-input images.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--max-valid-batches-loss", type=int, default=100)
    parser.add_argument("--max-valid-batches-generation", type=int, default=50)
    parser.add_argument("--teacher-forcing-error-rate", type=float, default=0.1)
    parser.add_argument("--compute-valid-loss", action="store_true")
    parser.add_argument("--compute-teds", action="store_true")
    parser.add_argument("--metric-for-best", default="valid_steds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = {
        "encoder": TableSeqEncoderConfig(),
        "decoder": {"tokenizer_path": args.tokenizer_path, "max_length": args.max_length},
        "device": device,
    }
    if args.checkpoint is not None:
        model_config["checkpoints_path"] = args.checkpoint

    model = TableSeqModel(model_config).to(device)
    tokenizer = model.decoder.tokenizer

    data_config = TableSeqDatasetConfig(
        dataset_root=args.dataset_root,
        label_file=args.label_file,
        train_batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        resize_factor=args.resize_factor,
        coordinate_labels_in_original_image_space=not args.labels_already_resized,
        normalize=True,
        image_padding_value=0.0,
    )
    train_loader, valid_loader = build_tableseq_dataloaders(data_config, tokenizer)

    train_config = TableSeqTrainingConfig(
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        lr=args.lr,
        teacher_forcing_error_rate=args.teacher_forcing_error_rate,
        token_substitution_path=args.token_substitution_json,
        save_every_n_steps=args.save_every,
        eval_every_n_steps=args.eval_every,
        generate_samples_every_n_steps=args.sample_every,
        max_valid_batches_for_loss=args.max_valid_batches_loss,
        max_valid_batches_for_generation=args.max_valid_batches_generation,
        compute_valid_loss=args.compute_valid_loss,
        compute_valid_struct_seq=True,
        compute_valid_steds=True,
        compute_valid_teds=args.compute_teds,
        compute_valid_box_metrics=True,
        metric_for_best=args.metric_for_best,
    )

    trainer = TableSeqTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=train_config,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
