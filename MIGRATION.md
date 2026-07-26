# Coordinate and validation migration notes

## Fixed coordinate convention

TableSeq now has one coordinate quantum: one `<x_i>` or `<y_i>` unit always
represents **5 pixels in the image presented to the model**. There is no
coordinate-quantum command-line option.

When labels are stored in original-image space, the dataset loader rescales
coordinate token IDs to the actual model-input dimensions. Generated boxes are
converted back to original-image pixels using the effective horizontal and
vertical image scales.

## Legacy labels

The default assumes coordinate tokens are expressed in original-image space.
Use `--labels-already-resized` only for a legacy label file whose coordinates
already correspond to resized model inputs.

## Standalone validation

`scripts/eval_tableseq.py` now exposes the same non-loss validation metrics used
by training: structure sequence, S-TEDS, optional TEDS, generated box IoU/F1,
box counts, coordinate L1 in original-image pixels, generation length, and
teacher-forced token/coordinate-token accuracies.
