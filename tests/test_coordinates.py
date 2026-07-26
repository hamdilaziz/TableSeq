import numpy as np

from tableseq.coordinates import (
    ImageGeometry,
    quantized_boxes_to_original_pixels,
    resolve_resize_factor,
    scale_coordinate_tokens,
)


def test_scale_coordinate_tokens_with_resize_factor() -> None:
    raw = "<td><x_10><y_20>A<x_30><y_40></td>"
    assert scale_coordinate_tokens(raw, scale_x=1.5, scale_y=1.5) == (
        "<td><x_15><y_30>A<x_45><y_60></td>"
    )


def test_scale_coordinate_tokens_uses_axis_specific_effective_scales() -> None:
    raw = "<x_10><y_10><x_20><y_20>"
    assert scale_coordinate_tokens(raw, scale_x=1.5, scale_y=2.0) == "<x_15><y_20><x_30><y_40>"


def test_quantized_boxes_convert_back_to_original_pixels() -> None:
    boxes = np.asarray([[0, 0, 30, 15]], dtype=np.float32)
    pixels = quantized_boxes_to_original_pixels(boxes, scale_x=1.5, scale_y=1.5)
    np.testing.assert_allclose(pixels, [[0, 0, 100, 50]])


def test_image_geometry_tracks_effective_scale() -> None:
    geometry = ImageGeometry(original_width=101, original_height=51, model_width=151, model_height=76)
    assert geometry.scale_x == 151 / 101
    assert geometry.scale_y == 76 / 51


def test_resize_factor_is_loaded_from_checkpoint_metadata() -> None:
    factor, source = resolve_resize_factor(
        requested=None,
        checkpoint_metadata={"data_config": {"resize_factor": 1.5}},
    )
    assert factor == 1.5
    assert source == "checkpoint"


def test_box_l1_is_reported_in_original_pixels_across_resize_factors() -> None:
    from tableseq.training.box_metrics import compute_box_metrics_from_strings

    # One-token error at 1.5x equals 5 / 1.5 original pixels.
    result = compute_box_metrics_from_strings(
        reference="<x_0><y_0><x_30><y_15>",
        prediction="<x_0><y_0><x_31><y_15>",
        image_scale_x=1.5,
        image_scale_y=1.5,
    )
    assert result.box_l1 is not None
    assert abs(result.box_l1 - (5.0 / 1.5) / 4.0) < 1e-5
