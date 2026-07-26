"""Coordinate conventions shared by TableSeq training, evaluation, and inference.

TableSeq stores coordinates as integer ``<x_i>`` / ``<y_i>`` tokens. One token
unit always represents five pixels in the image resolution seen by the model.
When an original image is resized before the encoder, label tokens are scaled to
the model-image resolution and generated tokens are divided by the effective
image scale when they are rendered over the original image.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

COORDINATE_QUANTUM_PX = 5.0
COORD_TOKEN_RE = re.compile(r"<([xy])_(\d+)>")


@dataclass(frozen=True)
class ImageGeometry:
    """Original and model-input image geometry."""

    original_width: int
    original_height: int
    model_width: int
    model_height: int

    @property
    def scale_x(self) -> float:
        return float(self.model_width) / float(self.original_width)

    @property
    def scale_y(self) -> float:
        return float(self.model_height) / float(self.original_height)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "original_width": int(self.original_width),
            "original_height": int(self.original_height),
            "model_width": int(self.model_width),
            "model_height": int(self.model_height),
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
        }


def validate_resize_factor(resize_factor: float) -> float:
    value = float(resize_factor)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"resize_factor must be a finite positive number, got {resize_factor!r}.")
    return value


def _round_nonnegative(value: float) -> int:
    """Round a non-negative coordinate using conventional half-up rounding."""
    return max(0, int(math.floor(float(value) + 0.5)))


def extract_coord_tokens(text: str) -> list[tuple[str, int]]:
    """Extract coordinate tokens as ordered ``(axis, value)`` pairs."""
    return [(match.group(1), int(match.group(2))) for match in COORD_TOKEN_RE.finditer(str(text))]


def extract_quantized_boxes(text: str) -> np.ndarray:
    """Extract valid ``x,y,x,y`` boxes and resynchronize after malformed tokens."""
    coords = extract_coord_tokens(text)
    boxes: list[list[float]] = []
    i = 0
    while i + 3 < len(coords):
        group = coords[i : i + 4]
        if [axis for axis, _ in group] == ["x", "y", "x", "y"]:
            boxes.append([float(value) for _, value in group])
            i += 4
        else:
            i += 1
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def scale_coordinate_tokens(text: str, scale_x: float, scale_y: float) -> str:
    """Scale coordinate tokens from original-image to model-image space.

    If an original coordinate is represented by ``token * 5`` pixels, resizing
    the image by ``scale`` gives ``token * 5 * scale`` pixels. Quantizing again
    by the same fixed five-pixel quantum therefore reduces to ``token * scale``.
    Axis-specific scales are used to account for integer resize rounding and
    optional maximum image dimensions.
    """
    scale_x = validate_resize_factor(scale_x)
    scale_y = validate_resize_factor(scale_y)

    if abs(scale_x - 1.0) < 1e-12 and abs(scale_y - 1.0) < 1e-12:
        return str(text)

    def replace(match: re.Match[str]) -> str:
        axis = match.group(1)
        value = int(match.group(2))
        scale = scale_x if axis == "x" else scale_y
        return f"<{axis}_{_round_nonnegative(value * scale)}>"

    return COORD_TOKEN_RE.sub(replace, str(text))


def quantized_boxes_to_original_pixels(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Convert quantized model-space boxes to original-image pixels."""
    scale_x = validate_resize_factor(scale_x)
    scale_y = validate_resize_factor(scale_y)

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    out = boxes.reshape(-1, 4).astype(np.float32).copy()
    out[:, [0, 2]] *= float(COORDINATE_QUANTUM_PX / scale_x)
    out[:, [1, 3]] *= float(COORDINATE_QUANTUM_PX / scale_y)
    return out


def checkpoint_resize_factor(metadata: dict | None) -> float | None:
    """Read a saved resize factor from TableSeq checkpoint metadata."""
    if not metadata:
        return None

    data_config = metadata.get("data_config")
    if isinstance(data_config, dict) and data_config.get("resize_factor") is not None:
        return validate_resize_factor(float(data_config["resize_factor"]))

    if metadata.get("resize_factor") is not None:
        return validate_resize_factor(float(metadata["resize_factor"]))

    return None


def resolve_resize_factor(
    requested: float | None,
    checkpoint_metadata: dict | None = None,
    fallback: float = 1.0,
) -> tuple[float, str]:
    """Resolve preprocessing scale from CLI input, checkpoint metadata, or fallback."""
    saved = checkpoint_resize_factor(checkpoint_metadata)

    if requested is not None:
        requested_value = validate_resize_factor(requested)
        if saved is not None and abs(requested_value - saved) > 1e-9:
            return requested_value, "explicit_override"
        return requested_value, "explicit"

    if saved is not None:
        return saved, "checkpoint"

    return validate_resize_factor(fallback), "fallback"
