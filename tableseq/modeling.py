from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .models.decoder import TableSeqDecoder
from .models.encoder import TableSeqEncoder

__all__ = [
    "TableSeqModel",
    "build_decoder_input_ids",
    "build_encoder_attention_mask",
    "infer_encoder_grid_size",
    "extract_sequences",
    "strip_generated_sequences",
]


def build_decoder_input_ids(
    batch_size: int,
    start_token_id: int,
    pad_token_id: int,
    device: torch.device,
    token_prompts: Optional[Sequence[Sequence[int] | Tensor]] = None,
) -> tuple[Tensor, list[int]]:
    """Build the TableSeq decoder prefix: ``<s>`` followed by an optional prompt."""
    if token_prompts is None:
        input_ids = torch.full((batch_size, 1), int(start_token_id), dtype=torch.long, device=device)
        return input_ids, [0 for _ in range(batch_size)]

    if len(token_prompts) != batch_size:
        raise ValueError(f"Expected {batch_size} token prompts, received {len(token_prompts)}.")

    prompts: list[list[int]] = []
    for prompt in token_prompts:
        if torch.is_tensor(prompt):
            prompt = prompt.detach().cpu().tolist()
        prompts.append([int(tok) for tok in prompt])

    prompt_lengths = [len(prompt) for prompt in prompts]
    max_prompt_len = max(prompt_lengths, default=0)
    rows = []
    for prompt in prompts:
        rows.append([int(pad_token_id)] * (max_prompt_len - len(prompt)) + [int(start_token_id)] + prompt)
    return torch.tensor(rows, dtype=torch.long, device=device), prompt_lengths


def build_encoder_attention_mask(
    image_attention_mask: Optional[Tensor],
    feature_height: int,
    feature_width: int,
) -> Optional[Tensor]:
    """Downsample a pixel-level image mask to the encoder token grid.

    Args:
        image_attention_mask: Tensor with shape ``(B, H, W)`` or ``(B, 1, H, W)``.
            Valid pixels must be 1 and padded pixels 0.
        feature_height, feature_width: spatial size of the encoder feature map.

    Returns:
        Tensor of shape ``(B, feature_height * feature_width)`` with 1 for valid
        visual tokens and 0 for padded tokens. The mask can be passed as
        ``encoder_attention_mask`` to the decoder.
    """
    if image_attention_mask is None:
        return None

    if image_attention_mask.dim() == 3:
        mask = image_attention_mask[:, None].float()
    elif image_attention_mask.dim() == 4:
        mask = image_attention_mask.float()
    else:
        raise ValueError(
            "image_attention_mask must have shape (B,H,W) or (B,1,H,W), "
            f"got {tuple(image_attention_mask.shape)}."
        )

    pooled = F.interpolate(mask, size=(int(feature_height), int(feature_width)), mode="nearest")
    return pooled[:, 0].flatten(1).long()



def infer_encoder_grid_size(
    encoder_hidden_states: Tensor,
    structure_logits: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Infer the 2D encoder grid size corresponding to decoder keys.

    The TableSeq structure head upsamples the encoder feature map along the
    vertical axis with a ConvTranspose2d(stride=(2, 1)). Therefore
    ``structure_logits.shape[-2:]`` is not always the same grid that is fed to
    the decoder. The decoder attends to ``encoder_hidden_states`` whose length
    is ``H_f * W_f``.
    """
    if encoder_hidden_states.dim() == 4:
        return int(encoder_hidden_states.shape[-2]), int(encoder_hidden_states.shape[-1])

    if encoder_hidden_states.dim() != 3:
        raise ValueError(
            "encoder_hidden_states must have shape (B,N,C) or (B,C,H,W), "
            f"got {tuple(encoder_hidden_states.shape)}."
        )

    token_len = int(encoder_hidden_states.shape[1])

    if structure_logits is not None:
        sh, sw = int(structure_logits.shape[-2]), int(structure_logits.shape[-1])

        if sh * sw == token_len:
            return sh, sw

        # Current TableSeq architecture: the structure head upsamples height only.
        if sh % 2 == 0 and (sh // 2) * sw == token_len:
            return sh // 2, sw

        # Defensive fallback in case a future structure head upsamples width only.
        if sw % 2 == 0 and sh * (sw // 2) == token_len:
            return sh, sw // 2

        # Defensive fallback in case both axes are upsampled.
        if sh % 2 == 0 and sw % 2 == 0 and (sh // 2) * (sw // 2) == token_len:
            return sh // 2, sw // 2

    raise ValueError(
        "Could not infer encoder grid size for the attention mask. "
        f"encoder token length={token_len}, "
        f"structure_logits shape={None if structure_logits is None else tuple(structure_logits.shape)}."
    )


def extract_sequences(generate_output: Tensor | Any) -> Tensor:
    if torch.is_tensor(generate_output):
        return generate_output
    if isinstance(generate_output, dict) and "sequences" in generate_output:
        return generate_output["sequences"]
    if hasattr(generate_output, "sequences"):
        return generate_output.sequences
    raise TypeError("Generation output must be a tensor, a dict with 'sequences', or an object with .sequences.")


def strip_generated_sequences(
    sequences: Tensor,
    input_ids: Tensor,
    prompt_lengths: Sequence[int],
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
) -> list[Tensor]:
    """Remove left padding and the ``<s> + prompt`` prefix from generated sequences."""
    outputs: list[Tensor] = []
    seq_cpu = sequences.detach().cpu()
    inp_cpu = input_ids.detach().cpu()

    for i, seq in enumerate(seq_cpu):
        seq_list = [int(tok) for tok in seq.tolist()]
        prefix_list = [int(tok) for tok in inp_cpu[i].tolist()]

        while prefix_list and prefix_list[0] == int(pad_token_id):
            prefix_list.pop(0)
        while seq_list and seq_list[0] == int(pad_token_id):
            seq_list.pop(0)

        prefix_len = 1 + int(prompt_lengths[i])
        seq_list = seq_list[prefix_len:]

        if eos_token_id is not None and int(eos_token_id) in seq_list:
            seq_list = seq_list[: seq_list.index(int(eos_token_id))]

        seq_list = [tok for tok in seq_list if tok != int(pad_token_id)]
        outputs.append(torch.tensor(seq_list, dtype=torch.long))

    return outputs


class TableSeqModel(nn.Module):
    """End-to-end TableSeq model: visual encoder + mBART decoder."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_metadata: dict[str, Any] = {}
        self.encoder = TableSeqEncoder(config.get("encoder"))
        self.decoder = TableSeqDecoder(config["decoder"])

        checkpoint_path = config.get("checkpoints_path")
        strict = bool(config.get("strict_load", True))

        if checkpoint_path is not None:
            info = self.load_pretrained_weights(checkpoint_path, strict=strict)
            self.checkpoint_metadata = dict(info.get("metadata", {}))

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint produced by :class:`TableSeqTrainer`."""
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
        required = {"encoder_state_dict", "decoder_state_dict"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(
                f"Invalid TableSeq checkpoint {checkpoint_path}: missing {sorted(missing)}. "
                "Use a checkpoint produced by TableSeqTrainer."
            )

        encoder_info = self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
        decoder_info = self.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=strict)
        metadata = {
            key: checkpoint[key]
            for key in (
                "data_config",
                "training_config",
                "global_step",
                "epoch",
                "best_metric",
                "best_metric_name",
            )
            if key in checkpoint
        }
        return {"encoder": encoder_info, "decoder": decoder_info, "metadata": metadata}

    def encode(self, images: Tensor, return_struct: bool = False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        return self.encoder(images, return_struct=return_struct)

    def encode_for_decoder(
        self,
        images: Tensor,
        image_attention_mask: Optional[Tensor] = None,
        use_structure_bias: bool = False,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Encode images and build the optional visual-token attention mask."""
        encoder_key_bias = None
        encoder_attention_mask = None

        # Use return_struct=True whenever a visual mask is needed, because the
        # structure logits expose the exact feature-grid size after the CNN.
        needs_feature_shape = image_attention_mask is not None
        enc = self.encoder(images, return_struct=bool(use_structure_bias or needs_feature_shape))

        if isinstance(enc, tuple):
            encoder_hidden_states = enc[0]
            structure_logits = enc[1]
            if use_structure_bias and len(enc) >= 3:
                encoder_key_bias = enc[2]
            feature_height, feature_width = infer_encoder_grid_size(
                encoder_hidden_states=encoder_hidden_states,
                structure_logits=structure_logits,
            )
            encoder_attention_mask = build_encoder_attention_mask(
                image_attention_mask,
                feature_height=feature_height,
                feature_width=feature_width,
            )
        else:
            encoder_hidden_states = enc

        if encoder_hidden_states.dim() == 4:
            encoder_hidden_states = encoder_hidden_states.flatten(2).transpose(1, 2).contiguous()
        if images.device.type != "cuda":
            encoder_hidden_states = encoder_hidden_states.float()
            if encoder_key_bias is not None:
                encoder_key_bias = encoder_key_bias.float()
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=images.device)
        return encoder_hidden_states, encoder_key_bias, encoder_attention_mask

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        use_structure_bias: bool = False,
        **kwargs: Any,
    ) -> Any:
        encoder_hidden_states, encoder_key_bias, inferred_encoder_attention_mask = self.encode_for_decoder(
            images,
            image_attention_mask=image_attention_mask,
            use_structure_bias=use_structure_bias,
        )
        if encoder_attention_mask is None:
            encoder_attention_mask = inferred_encoder_attention_mask
        return self.decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_key_bias=encoder_key_bias,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        input_ids: Tensor,
        image_attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        use_structure_bias: bool = False,
        max_length: Optional[int] = None,
        early_stopping: bool = True,
        num_beams: int = 1,
        do_sample: bool = False,
        use_cache: bool = True,
        bad_words_ids: Optional[list[list[int]]] = None,
        return_dict_in_generate: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **generate_kwargs: Any,
    ) -> Tensor | Any:
        del early_stopping
        self.eval()
        tokenizer = self.decoder.tokenizer
        if max_length is None:
            max_length = int(self.decoder.model.config.max_position_embeddings)
        if bad_words_ids is None:
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            if unk_token_id is not None:
                bad_words_ids = [[int(unk_token_id)]]

        encoder_hidden_states, encoder_key_bias, inferred_encoder_attention_mask = self.encode_for_decoder(
            images,
            image_attention_mask=image_attention_mask,
            use_structure_bias=use_structure_bias,
        )
        if encoder_attention_mask is None:
            encoder_attention_mask = inferred_encoder_attention_mask
        return self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_key_bias=encoder_key_bias,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
            num_beams=num_beams,
            do_sample=do_sample,
            bad_words_ids=bad_words_ids,
            return_dict_in_generate=return_dict_in_generate,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

    def build_input_ids_from_prompt(
        self,
        prompt: Optional[str] = "<html>",
        batch_size: int = 1,
        device: Optional[str | torch.device] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        tokenizer = self.decoder.tokenizer
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)

        start_token_id = getattr(self.decoder.model.config, "decoder_start_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(tokenizer, "bos_token_id", None)
        if start_token_id is None:
            raise ValueError("No decoder_start_token_id or bos_token_id is defined.")

        if prompt is None or prompt == "":
            return build_decoder_input_ids(
                batch_size,
                int(start_token_id),
                int(tokenizer.pad_token_id),
                device,
                None,
            )

        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        token_prompts = [prompt_ids.clone() for _ in range(batch_size)]
        return build_decoder_input_ids(
            batch_size,
            int(start_token_id),
            int(tokenizer.pad_token_id),
            device,
            token_prompts,
        )

    @torch.no_grad()
    def predict(
        self,
        batch_data: dict[str, Any],
        max_length: Optional[int] = None,
        start_token_id: Optional[int] = None,
        use_amp: bool = False,
        use_structure_bias: bool = False,
        num_beams: int = 1,
        do_sample: bool = False,
        use_cache: bool = True,
        return_dict_in_generate: bool = True,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        images = batch_data.get("imgs", batch_data.get("images"))
        if images is None:
            raise KeyError("batch_data must contain 'imgs' or 'images'.")
        device = next(self.parameters()).device
        images = images.to(device)
        image_attention_mask = batch_data.get("image_attention_mask", batch_data.get("imgs_mask"))
        if image_attention_mask is not None:
            image_attention_mask = image_attention_mask.to(device)

        tokenizer = self.decoder.tokenizer
        if start_token_id is None:
            start_token_id = getattr(self.decoder.model.config, "decoder_start_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(tokenizer, "bos_token_id", None)
        if start_token_id is None:
            raise ValueError("No decoder_start_token_id or bos_token_id is defined.")

        token_prompts = batch_data.get("token_prompt")
        input_ids, prompt_lengths = build_decoder_input_ids(
            batch_size=images.size(0),
            start_token_id=int(start_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            token_prompts=token_prompts,
        )

        amp_enabled = bool(use_amp and device.type == "cuda")
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
            generated = self.generate(
                images=images,
                input_ids=input_ids,
                image_attention_mask=image_attention_mask,
                use_structure_bias=use_structure_bias,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                use_cache=use_cache,
                return_dict_in_generate=return_dict_in_generate,
                **generate_kwargs,
            )

        sequences = extract_sequences(generated)
        token_x = strip_generated_sequences(
            sequences=sequences,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        str_x = [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in token_x]

        out: dict[str, Any] = {
            "str_x": str_x,
            "token_x": token_x,
            "sequences": sequences,
            "input_ids": input_ids,
        }
        if "raw_labels" in batch_data:
            out["str_y"] = batch_data["raw_labels"]
        return out
