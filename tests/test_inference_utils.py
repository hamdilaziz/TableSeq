from pathlib import Path

from PIL import Image

from tableseq.coordinates import ImageGeometry
from tableseq.inference import (
    clean_prediction,
    draw_predicted_boxes,
    extract_predicted_boxes,
    save_html_report,
    save_overlay_image,
)


def test_extract_boxes_uses_fixed_quantum_at_original_scale() -> None:
    raw = "<td><x_0><y_0>A<x_20><y_10></td><td><x_20><y_0>B<x_40><y_10></td>"
    boxes = extract_predicted_boxes(raw)

    assert len(boxes) == 2
    assert boxes[0].quantized == [0.0, 0.0, 20.0, 10.0]
    assert boxes[0].pixel == [0.0, 0.0, 100.0, 50.0]
    assert boxes[1].pixel == [100.0, 0.0, 200.0, 50.0]


def test_extract_boxes_maps_resized_model_coordinates_back_to_original_image() -> None:
    raw = "<td><x_0><y_0>A<x_30><y_15></td>"
    boxes = extract_predicted_boxes(raw, image_scale_x=1.5, image_scale_y=1.5)

    assert len(boxes) == 1
    assert boxes[0].pixel == [0.0, 0.0, 100.0, 50.0]


def test_extract_boxes_resynchronizes_after_malformed_coordinates() -> None:
    raw = "<x_99><td><x_0><y_0>A<x_20><y_10></td><x_88>"
    boxes = extract_predicted_boxes(raw)

    assert len(boxes) == 1
    assert boxes[0].pixel == [0.0, 0.0, 100.0, 50.0]


def test_extract_boxes_normalizes_reversed_corners_for_visualization() -> None:
    boxes = extract_predicted_boxes("<x_20><y_10>A<x_0><y_0>")
    assert boxes[0].quantized == [20.0, 10.0, 0.0, 0.0]
    assert boxes[0].pixel == [0.0, 0.0, 100.0, 50.0]


def test_clean_prediction_matches_eval_style() -> None:
    raw = "<s><html><table><tr><td><x_0><y_0>A<x_20><y_10></td></tr></table></s>"
    assert clean_prediction(raw) == "<table><tr><td>A</td></tr></table>"


def test_save_inference_report_and_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "table.png"
    Image.new("RGB", (100, 50), "white").save(image_path)
    raw = "<s><html><table><tr><td><x_0><y_0>A<x_20><y_10></td></tr></table></s>"
    boxes = extract_predicted_boxes(raw)
    geometry = ImageGeometry(100, 50, 150, 75)

    overlay_path = tmp_path / "overlay.png"
    report_path = tmp_path / "report.html"
    save_overlay_image(image_path, boxes, overlay_path)
    save_html_report(
        image_path,
        raw,
        "<table><tr><td>A</td></tr></table>",
        boxes,
        report_path,
        geometry=geometry,
    )

    assert overlay_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "fixed 5-pixel quantum" in report
    assert "x × 5 / 1.5" in report


def test_draw_boxes_returns_image_copy() -> None:
    image = Image.new("RGB", (100, 50), "white")
    boxes = extract_predicted_boxes("<x_0><y_0>A<x_20><y_10>")
    out = draw_predicted_boxes(image, boxes)

    assert out is not image
    assert out.size == image.size
