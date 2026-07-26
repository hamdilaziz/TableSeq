"""Single-image inference and visualization for TableSeq."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .coordinates import (
    COORDINATE_QUANTUM_PX,
    ImageGeometry,
    extract_quantized_boxes,
    quantized_boxes_to_original_pixels,
    resolve_resize_factor,
)
from .data import TableSeqDatasetConfig, load_image_for_model
from .modeling import TableSeqModel, extract_sequences
from .models.encoder import TableSeqEncoderConfig
from .training.metrics import clean_tableseq_html


ALLOWED_TABLE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption"}
ALLOWED_CELL_ATTRS = {"rowspan", "colspan"}


@dataclass(frozen=True)
class PredictedBox:
    """One generated box in both token and pixel coordinates."""

    index: int
    quantized: list[float]
    pixel: list[float]


def clean_prediction(value: str) -> str:
    """Return generated table HTML without control or coordinate tokens."""
    return clean_tableseq_html(value)


def preprocess_image(
    image_path: str | Path,
    resize_factor: float = 1.0,
    normalize: bool = True,
) -> tuple[torch.FloatTensor, torch.LongTensor, ImageGeometry]:
    """Load one image through the exact dataset preprocessing path."""
    cfg = TableSeqDatasetConfig(
        dataset_root=".",
        resize_factor=resize_factor,
        normalize=normalize,
    )
    arr, geometry = load_image_for_model(image_path, cfg)
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).contiguous()
    mask = torch.ones((1, arr.shape[0], arr.shape[1]), dtype=torch.long)
    return tensor, mask, geometry


def extract_predicted_boxes(
    raw_prediction: str,
    image_scale_x: float = 1.0,
    image_scale_y: float | None = None,
) -> list[PredictedBox]:
    """Extract valid ``x,y,x,y`` groups and map them to original pixels.

    Coordinate tokens always represent five pixels in the model-input image.
    Rendering over the original image therefore uses ``token * 5 / scale``.
    The axis-aware parser resynchronizes after malformed generated tokens.
    """
    if image_scale_y is None:
        image_scale_y = image_scale_x

    boxes_quantized = extract_quantized_boxes(raw_prediction)
    boxes_pixels = quantized_boxes_to_original_pixels(
        boxes_quantized,
        scale_x=image_scale_x,
        scale_y=image_scale_y,
    )
    if len(boxes_pixels):
        x1 = np.minimum(boxes_pixels[:, 0], boxes_pixels[:, 2])
        y1 = np.minimum(boxes_pixels[:, 1], boxes_pixels[:, 3])
        x2 = np.maximum(boxes_pixels[:, 0], boxes_pixels[:, 2])
        y2 = np.maximum(boxes_pixels[:, 1], boxes_pixels[:, 3])
        boxes_pixels = np.stack([x1, y1, x2, y2], axis=1)

    return [
        PredictedBox(
            index=i + 1,
            quantized=[float(v) for v in boxes_quantized[i].tolist()],
            pixel=[float(v) for v in boxes_pixels[i].tolist()],
        )
        for i in range(len(boxes_pixels))
    ]


def draw_predicted_boxes(
    image: Image.Image,
    boxes: Iterable[PredictedBox],
    color: str = "red",
    width: int = 2,
    draw_labels: bool = False,
) -> Image.Image:
    """Draw predicted boxes over an image in original-image coordinates."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        x1, y1, x2, y2 = box.pixel
        draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=color, width=width)
        if draw_labels:
            label = str(box.index)
            bbox = draw.textbbox((float(x1), float(y1)), label, font=font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1, x1 + label_w + 4, y1 + label_h + 4], fill=color)
            draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

    return out


def save_overlay_image(
    image_path: str | Path,
    boxes: Iterable[PredictedBox],
    output_path: str | Path,
    draw_labels: bool = False,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw_predicted_boxes(image, boxes, draw_labels=draw_labels).save(output_path)


class _SafeTableHTMLParser(HTMLParser):
    """Conservative table-only sanitizer for the local HTML report."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TABLE_TAGS:
            return
        safe_attrs = []
        if tag in {"td", "th"}:
            for key, value in attrs:
                key = key.lower()
                if key not in ALLOWED_CELL_ATTRS or value is None:
                    continue
                value = str(value).strip().strip('"\'')
                if value.isdigit() and 1 <= int(value) <= 100:
                    safe_attrs.append(f'{key}="{html.escape(value, quote=True)}"')
        attr_text = "" if not safe_attrs else " " + " ".join(safe_attrs)
        self.parts.append(f"<{tag}{attr_text}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ALLOWED_TABLE_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{html.escape(name)};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{html.escape(name)};")

    def get_html(self) -> str:
        return "".join(self.parts).strip()


def sanitize_table_html(table_html: str) -> str:
    cleaned = clean_prediction(table_html)
    parser = _SafeTableHTMLParser()
    parser.feed(cleaned)
    sanitized = parser.get_html()
    return sanitized or "<p>No valid table HTML was generated.</p>"


def _image_to_base64_data_uri(path: str | Path) -> tuple[str, int, int]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}", width, height


def _build_svg_overlay(boxes: Iterable[PredictedBox], width: int, height: int) -> str:
    rects: list[str] = []
    for box in boxes:
        x1, y1, x2, y2 = box.pixel
        rects.append(
            '<rect '
            f'x="{x1:.2f}" y="{y1:.2f}" width="{x2 - x1:.2f}" height="{y2 - y1:.2f}" '
            'fill="none" stroke="red" stroke-width="2" />'
        )
    return f'<svg viewBox="0 0 {width} {height}" aria-label="Predicted boxes">' + "".join(rects) + "</svg>"


def save_html_report(
    image_path: str | Path,
    raw_prediction: str,
    cleaned_html: str,
    boxes: list[PredictedBox],
    output_path: str | Path,
    geometry: ImageGeometry,
) -> None:
    """Save a self-contained visual report for one TableSeq prediction."""
    image_uri, width, height = _image_to_base64_data_uri(image_path)
    safe_table = sanitize_table_html(cleaned_html)
    svg_overlay = _build_svg_overlay(boxes, width, height)
    raw_escaped = html.escape(raw_prediction)
    clean_escaped = html.escape(cleaned_html)

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TableSeq inference report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52606d; margin-top: 0; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
      gap: 24px;
      align-items: start;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .image-wrap {{ position: relative; display: inline-block; max-width: 100%; }}
    .image-wrap img {{ display: block; max-width: 100%; height: auto; }}
    .image-wrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #9fb3c8; padding: 6px 8px; vertical-align: top; }}
    thead td, thead th {{ background: #eef2f7; font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #111827;
      color: #e5e7eb;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
      max-height: 420px;
      overflow: auto;
    }}
    .small {{ color: #627d98; font-size: 13px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>TableSeq inference report</h1>
  <p class="meta">
    Image: {html.escape(str(image_path))} · boxes: {len(boxes)} ·
    model size: {geometry.model_width}×{geometry.model_height} ·
    original size: {geometry.original_width}×{geometry.original_height}
  </p>
  <section class="grid">
    <div class="card">
      <h2>Input image with predicted boxes</h2>
      <div class="image-wrap">
        <img src="{image_uri}" width="{width}" height="{height}" alt="Input table image" />
        {svg_overlay}
      </div>
      <p class="small">
        Coordinate tokens use a fixed {COORDINATE_QUANTUM_PX:g}-pixel quantum in model-image space.
        Original-image boxes use
        <code>x × {COORDINATE_QUANTUM_PX:g} / {geometry.scale_x:.6g}</code> and
        <code>y × {COORDINATE_QUANTUM_PX:g} / {geometry.scale_y:.6g}</code>.
      </p>
    </div>
    <div class="card">
      <h2>Rendered predicted HTML</h2>
      {safe_table}
    </div>
  </section>
  <section class="card" style="margin-top: 24px;">
    <h2>Cleaned HTML</h2>
    <pre>{clean_escaped}</pre>
  </section>
  <section class="card" style="margin-top: 24px;">
    <h2>Raw generated sequence</h2>
    <pre>{raw_escaped}</pre>
  </section>
</body>
</html>
"""
    Path(output_path).write_text(report, encoding="utf-8")


@torch.no_grad()
def infer_one_image(
    checkpoint: str | Path,
    tokenizer: str | Path,
    image: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    max_length: int = 2048,
    resize_factor: float | None = None,
    strict_load: bool = True,
) -> dict[str, Any]:
    """Run TableSeq on one image and save prediction artifacts."""
    image = Path(image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model_config = {
        "encoder": TableSeqEncoderConfig(),
        "decoder": {
            "tokenizer_path": str(tokenizer),
            "max_length": int(max_length),
        },
        "checkpoints_path": str(checkpoint),
        "strict_load": bool(strict_load),
        "device": str(runtime_device),
    }
    model = TableSeqModel(model_config).to(runtime_device)
    model.eval()

    resize_factor, resize_source = resolve_resize_factor(
        requested=resize_factor,
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

    images, image_attention_mask, geometry = preprocess_image(
        image,
        resize_factor=resize_factor,
        normalize=True,
    )
    images = images.to(runtime_device, non_blocking=True)
    image_attention_mask = image_attention_mask.to(runtime_device, non_blocking=True)

    input_ids, _ = model.build_input_ids_from_prompt("<html>", batch_size=1, device=runtime_device)
    output = model.generate(
        images=images,
        input_ids=input_ids,
        image_attention_mask=image_attention_mask,
        max_length=max_length,
        use_cache=True,
        use_structure_bias=False,
        return_dict_in_generate=True,
    )
    sequences = extract_sequences(output)

    raw_prediction = model.decoder.tokenizer.decode(sequences[0], skip_special_tokens=False)
    cleaned_html = clean_prediction(raw_prediction)
    boxes = extract_predicted_boxes(
        raw_prediction,
        image_scale_x=geometry.scale_x,
        image_scale_y=geometry.scale_y,
    )

    raw_path = output_dir / "prediction_raw.txt"
    html_path = output_dir / "prediction.html"
    boxes_path = output_dir / "prediction_boxes.json"
    overlay_path = output_dir / "prediction_boxes.png"
    report_path = output_dir / "report.html"
    jsonl_path = output_dir / "prediction.jsonl"

    raw_path.write_text(raw_prediction, encoding="utf-8")
    html_path.write_text(sanitize_table_html(cleaned_html), encoding="utf-8")
    boxes_path.write_text(json.dumps([asdict(box) for box in boxes], ensure_ascii=False, indent=2), encoding="utf-8")
    save_overlay_image(image, boxes, overlay_path)
    save_html_report(
        image_path=image,
        raw_prediction=raw_prediction,
        cleaned_html=cleaned_html,
        boxes=boxes,
        output_path=report_path,
        geometry=geometry,
    )

    row = {
        "name": image.name,
        "path": str(image),
        "prediction_raw": raw_prediction,
        "prediction_clean": cleaned_html,
        "num_boxes": len(boxes),
        "resize_factor": resize_factor,
        "resize_factor_source": resize_source,
        "coordinate_quantum_px": COORDINATE_QUANTUM_PX,
        "image_geometry": geometry.to_dict(),
    }
    jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "raw_prediction": str(raw_path),
        "prediction_html": str(html_path),
        "prediction_boxes": str(boxes_path),
        "prediction_overlay": str(overlay_path),
        "prediction_jsonl": str(jsonl_path),
        "report": str(report_path),
        "num_boxes": len(boxes),
        "coordinate_quantum_px": COORDINATE_QUANTUM_PX,
        "resize_factor": resize_factor,
        "resize_factor_source": resize_source,
        "image_geometry": geometry.to_dict(),
        "device": str(runtime_device),
    }
