from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import pad

try:
    from torch.utils.checkpoint import checkpoint
except Exception:  # pragma: no cover
    checkpoint = None


__all__ = [
    "AnisotropicStructureHead",
    "ConvBlock",
    "DepthSepConv2D",
    "DSCBlock",
    "MixDropout",
    "PositionalEncoding2D",
    "FeatureUpdater",
    "TableSeqEncoder",
    "TableSeqEncoderConfig",
    "build_key_bias_from_structure",
]


def _autocast_disabled(device_type: str):
    """Return a context with autocast disabled when the backend supports it."""
    try:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    except Exception:
        if device_type == "cuda":
            try:
                return torch.cuda.amp.autocast(enabled=False)
            except Exception:
                pass
        return nullcontext()


def _run_block(block: nn.Module, x: Tensor, use_checkpointing: bool) -> Tensor:
    if use_checkpointing and checkpoint is not None and x.requires_grad:
        return checkpoint(block, x, use_reentrant=False)
    return block(x)


@dataclass(frozen=True)
class TableSeqEncoderConfig:
    """Configuration for :class:`TableSeqEncoder`."""

    dropout: float = 0.1
    in_channels: int = 3
    hidden_dim: int = 1024
    transformer_heads: int = 4
    transformer_layers: int = 1
    transformer_ff_dim: int = 2048
    use_checkpointing: bool = False
    use_2d_positional_encoding: bool = True
    structure_mid_channels: int = 256
    structure_dropout: float = 0.2
    structure_bias_lambda: float = 2.0
    detach_structure_bias: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None) -> "TableSeqEncoderConfig":
        if params is None:
            return cls()
        valid = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(params) - valid)
        if unknown:
            raise ValueError(f"Unknown TableSeqEncoderConfig fields: {unknown}")
        return cls(**dict(params))


class PositionalEncoding2D(nn.Module):
    """Dynamic fixed 2D sinusoidal encoding for feature maps of shape ``(B, C, H, W)``."""

    def __init__(self, dim: int, enabled: bool = True) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"2D positional encoding requires dim % 4 == 0, got {dim}.")
        self.dim = int(dim)
        self.enabled = bool(enabled)

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got shape {tuple(x.shape)}.")
        _, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}.")
        return x + self._build(height, width, x.device, x.dtype)

    def _build(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        half_dim = self.dim // 2
        pe = torch.zeros((1, self.dim, height, width), device=device, dtype=dtype)

        div = torch.exp(
            -torch.arange(0, half_dim, 2, device=device, dtype=torch.float32)
            / self.dim
            * math.log(10000.0)
        ).to(dtype=dtype)

        y = torch.arange(height, device=device, dtype=dtype)
        x = torch.arange(width, device=device, dtype=dtype)

        pe[:, :half_dim:2, :, :] = torch.sin(div[:, None] * y[None, :])[None, :, :, None]
        pe[:, 1:half_dim:2, :, :] = torch.cos(div[:, None] * y[None, :])[None, :, :, None]
        pe[:, half_dim::2, :, :] = torch.sin(div[:, None] * x[None, :])[None, :, None, :]
        pe[:, half_dim + 1::2, :, :] = torch.cos(div[:, None] * x[None, :])[None, :, None, :]
        return pe


class FeatureUpdater(nn.Module):
    """Compatibility wrapper for the original DAN ``FeaturesUpdater`` call site.

    The original TableSeq encoder calls ``feature_updater.get_pos_features(x)``
    before flattening the CNN map.  This class keeps the same API while using a
    deterministic 2D sinusoidal encoding, so it has no trainable weights and does
    not affect checkpoint keys.
    """

    def __init__(self, dim: int = 1024, enabled: bool = True) -> None:
        super().__init__()
        self.pe = PositionalEncoding2D(dim=dim, enabled=enabled)

    def get_pos_features(self, x: Tensor) -> Tensor:
        return self.pe(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.get_pos_features(x)


class MixDropout(nn.Module):
    """Randomly choose element-wise dropout or channel-wise dropout during training."""

    def __init__(self, dropout_proba: float = 0.4, dropout2d_proba: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_proba)
        self.dropout2d = nn.Dropout2d(dropout2d_proba)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        if torch.rand((), device=x.device) < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class ConvBlock(nn.Module):
    """Three-convolution block used in the convolutional stem."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-3, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout, dropout / 2)

    def forward(self, x: Tensor) -> Tensor:
        dropout_position = int(torch.randint(1, 4, (), device=x.device)) if self.training else 0

        x = self.activation(self.conv1(x))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm(x)
        x = self.activation(self.conv3(x))
        if dropout_position == 3:
            x = self.dropout(x)
        return x


class DepthSepConv2D(nn.Module):
    """Depthwise-separable 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] = (3, 3),
        activation: nn.Module | None = None,
        padding: bool | Tuple[int, int] = True,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must contain exactly two values.")
        kernel = (int(kernel_size[0]), int(kernel_size[1]))

        self.post_pad: Tuple[int, int, int, int] | None = None
        if padding is True:
            if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
                pad_h = kernel[0] - 1
                pad_w = kernel[1] - 1
                self.post_pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                conv_padding = (0, 0)
            else:
                conv_padding = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
        elif padding is False:
            conv_padding = (0, 0)
        else:
            conv_padding = padding

        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            dilation=dilation,
            stride=stride,
            padding=conv_padding,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.depth_conv(x)
        if self.post_pad is not None:
            x = pad(x, self.post_pad)
        if self.activation is not None:
            x = self.activation(x)
        return self.point_conv(x)


class DSCBlock(nn.Module):
    """Depthwise-separable block used in the encoder body."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        activation: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_channels, out_channels, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_channels, out_channels, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-3, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout, dropout / 2)

    def forward(self, x: Tensor) -> Tensor:
        dropout_position = int(torch.randint(1, 4, (), device=x.device)) if self.training else 0

        x = self.activation(self.conv1(x))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm(x)
        x = self.conv3(x)
        if dropout_position == 3:
            x = self.dropout(x)
        return x


class AnisotropicStructureHead(nn.Module):
    """Predict row-separator, column-separator and corner heatmaps from the visual feature map."""

    def __init__(self, in_channels: int, mid_channels: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 3, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def build_key_bias_from_structure(
    structure_logits: Tensor,
    feature_height: int,
    feature_width: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    lambda0: float = 2.0,
    use_entropy_scale: bool = True,
    detach_bias: bool = True,
    clamp_val: float = 4.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Convert structure heatmaps into a key-only cross-attention bias."""
    if structure_logits.dim() != 4 or structure_logits.shape[1] != 3:
        raise ValueError(
            "structure_logits must have shape (B, 3, H, W), "
            f"got {tuple(structure_logits.shape)}."
        )

    bsz = structure_logits.shape[0]
    probs = torch.sigmoid(structure_logits.float())

    resized = F.interpolate(probs, size=(feature_height, feature_width), mode="bilinear", align_corners=False)
    row_sep = resized[:, 0]
    col_sep = resized[:, 1]
    corners = resized[:, 2]

    row_profile = row_sep.max(dim=2).values[:, :, None].expand(bsz, feature_height, feature_width)
    col_profile = col_sep.max(dim=1).values[:, None, :].expand(bsz, feature_height, feature_width)
    bias_map = alpha * row_profile + beta * col_profile + gamma * corners

    bias = bias_map.flatten(1)
    bias = (bias - bias.mean(dim=1, keepdim=True)) / bias.std(dim=1, keepdim=True).clamp_min(eps)

    if use_entropy_scale:
        p = resized.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean(dim=(1, 2, 3), keepdim=False)
        confidence = 1.0 - entropy
        scale = lambda0 * confidence[:, None]
    else:
        scale = torch.full((bsz, 1), lambda0, device=bias.device, dtype=bias.dtype)

    bias = (scale * bias).clamp(-clamp_val, clamp_val).to(dtype=structure_logits.dtype)
    if detach_bias:
        bias = bias.detach()

    diagnostics = {
        "row_sep": row_sep,
        "col_sep": col_sep,
        "corners": corners,
        "row_profile": row_profile,
        "col_profile": col_profile,
        "bias_map": bias_map,
    }
    return bias, diagnostics


class TableSeqEncoder(nn.Module):
    """Visual encoder for TableSeq.

    The output sequence has shape ``(B, H_f * W_f, hidden_dim)``. With the default
    stride schedule, the feature grid is approximately ``H/16 x W/8``.
    """

    def __init__(self, params: TableSeqEncoderConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            params if isinstance(params, TableSeqEncoderConfig) else TableSeqEncoderConfig.from_mapping(params)
        )

        cfg = self.config
        self.stem = nn.Sequential(
            ConvBlock(cfg.in_channels, 32, stride=(1, 1), dropout=cfg.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=cfg.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=cfg.dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=cfg.dropout),
            ConvBlock(256, 512, stride=(2, 1), dropout=cfg.dropout),
            ConvBlock(512, 512, stride=(1, 1), dropout=cfg.dropout),
        )

        self.body = nn.Sequential(
            DSCBlock(512, 512, stride=(1, 1), dropout=cfg.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=cfg.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=cfg.dropout),
            DSCBlock(512, cfg.hidden_dim, stride=(1, 1), dropout=cfg.dropout),
        )

        self.structure_head = AnisotropicStructureHead(
            cfg.hidden_dim,
            mid_channels=cfg.structure_mid_channels,
            dropout=cfg.structure_dropout,
        )
        self.position = FeatureUpdater(cfg.hidden_dim, enabled=cfg.use_2d_positional_encoding)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)

    def forward(self, images: Tensor, return_struct: bool = False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if images.dim() != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}.")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} input channels, got {images.shape[1]}.")

        x = images
        for i, block in enumerate(self.stem):
            if i >= 5:
                with _autocast_disabled(x.device.type):
                    x = _run_block(block, x.float(), self.config.use_checkpointing)
            else:
                x = _run_block(block, x, self.config.use_checkpointing)

        with _autocast_disabled(x.device.type):
            x = x.float()
            for block in self.body:
                residual = x
                update = _run_block(block, x, self.config.use_checkpointing)
                x = residual + update if residual.shape == update.shape else update

            structure_logits = self.structure_head(x)
            x = self.position.get_pos_features(x)
            _, _, feature_height, feature_width = x.shape
            tokens = x.flatten(2).transpose(1, 2).contiguous()
            tokens = self.transformer(tokens)

        if return_struct:
            key_bias, _ = build_key_bias_from_structure(
                structure_logits,
                feature_height=feature_height,
                feature_width=feature_width,
                lambda0=self.config.structure_bias_lambda,
                detach_bias=self.config.detach_structure_bias,
            )
            return tokens, structure_logits, key_bias
        return tokens
