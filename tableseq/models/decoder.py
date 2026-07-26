from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from transformers import XLMRobertaTokenizerFast
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import BaseModelOutput
except Exception:  # pragma: no cover
    XLMRobertaTokenizerFast = None  # type: ignore
    BaseModelOutput = None  # type: ignore
    try:
        from transformers.utils import ModelOutput  # type: ignore
    except Exception:
        ModelOutput = dict  # type: ignore

try:
    from .modeling_mbart import MBartConfig, MBartForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    try:
        from transformers import MBartConfig, MBartForCausalLM  # type: ignore
    except Exception:  # pragma: no cover
        MBartConfig = None  # type: ignore
        MBartForCausalLM = None  # type: ignore


__all__ = ["TableSeqDecoder"]


class DecoderOutput(dict):
    """Small dict/attribute output compatible with the generation utilities."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self


class TableSeqDecoder(nn.Module):
    """mBART decoder used by TableSeq.

    The wrapper owns an ``MBartForCausalLM`` instance and uses a local greedy
    generation loop so cached decoding remains stable across supported
    Transformers versions.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.__name__ = "TableSeqDecoder"

        if XLMRobertaTokenizerFast is None or MBartConfig is None or MBartForCausalLM is None:
            raise ImportError(
                "TableSeqDecoder requires transformers and the local "
                "tableseq.models.modeling_mbart module."
            )

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(params["tokenizer_path"])
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.extra_vocab_slots = int(params.get("extra_vocab_slots", 1))

        base_vocab_size = int(len(self.tokenizer.get_vocab()))
        self.vocab_size = base_vocab_size + self.extra_vocab_slots

        config = MBartConfig(
            is_decoder=True,
            is_encoder_decoder=bool(params.get("is_encoder_decoder", True)),
            add_cross_attention=bool(params.get("add_cross_attention", True)),
            decoder_layers=int(params.get("num_layers", params.get("decoder_layer", 4))),
            decoder_attention_heads=int(params.get("num_heads", 16)),
            decoder_ffn_dim=int(params.get("ffn_dim", 4096)),
            d_model=int(params.get("d_model", 1024)),
            max_position_embeddings=int(params["max_length"]),
            vocab_size=self.vocab_size,
            scale_embedding=True,
            add_final_layer_norm=True,
            pad_token_id=self.pad_token_id,
            bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            decoder_start_token_id=getattr(
                self.tokenizer,
                "bos_token_id",
                getattr(self.tokenizer, "cls_token_id", self.pad_token_id),
            ),
        )

        self.model = MBartForCausalLM(config)
        self.model.model.decoder.embed_tokens.padding_idx = self.pad_token_id

        self.model.forward = self.forward
        self.model.config.is_encoder_decoder = True
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        extra_special_tokens = params.get("extra_special_tokens")
        if extra_special_tokens:
            self.add_special_tokens(extra_special_tokens)

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        unique_tokens = sorted(set(tokens))
        if not unique_tokens:
            return
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": unique_tokens})
        if num_added <= 0:
            return
        new_vocab_size = len(self.tokenizer) + self.extra_vocab_slots
        self.model.resize_token_embeddings(new_vocab_size)
        self.vocab_size = new_vocab_size
        self.model.config.vocab_size = new_vocab_size
        self.model.model.decoder.embed_tokens.padding_idx = self.pad_token_id

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        past: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[Tensor] = None,
        token_prompts: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_key_bias: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility method for Hugging Face generation.

        The local greedy loop does not rely on this method, but keeping it makes
        the wrapped mBART usable with ``self.model.generate`` when needed.
        """
        if past is not None and past_key_values is None:
            past_key_values = past

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:, :]

        encoder_hidden_states = None
        if encoder_outputs is not None:
            if torch.is_tensor(encoder_outputs):
                encoder_hidden_states = encoder_outputs
            elif hasattr(encoder_outputs, "last_hidden_state"):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, dict):
                encoder_hidden_states = encoder_outputs["last_hidden_state"]
            else:
                encoder_hidden_states = encoder_outputs[0]

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "token_prompts": token_prompts,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_key_bias": encoder_key_bias,
        }

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        labels: Optional[Tensor] = None,
        token_prompts: Optional[Tensor] = None,
        encoder_key_bias: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs: Any,
    ) -> Any:
        del token_prompts
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        decoder_kwargs = dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if encoder_key_bias is not None:
            decoder_kwargs["encoder_key_bias"] = encoder_key_bias

        outputs = self.model.model.decoder(**decoder_kwargs)
        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            loss_fn = nn.NLLLoss(ignore_index=self.pad_token_id)
            loss = loss_fn(log_probs.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return DecoderOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def _mask_bad_words(logits: Tensor, bad_words_ids: Optional[list[list[int]]]) -> Tensor:
        if not bad_words_ids:
            return logits
        for group in bad_words_ids:
            if len(group) == 1:
                token_id = int(group[0])
                if 0 <= token_id < logits.size(-1):
                    logits[:, token_id] = -float("inf")
        return logits

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_key_bias: Optional[Tensor] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        bad_words_ids: Optional[list[list[int]]] = None,
        return_dict_in_generate: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> Any:
        """Greedy autoregressive decoding with a working past-key-value cache."""
        if max_length is None:
            max_length = int(self.model.config.max_position_embeddings)
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        generated = input_ids.clone()
        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
        past_key_values = None
        current_input_ids = generated

        while generated.size(1) < int(max_length):
            attention_mask = generated.ne(int(pad_token_id)).long()
            outputs = self.forward(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_key_bias=encoder_key_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            logits = self._mask_bad_words(logits, bad_words_ids)
            next_tokens = torch.argmax(logits, dim=-1)

            if eos_token_id is not None:
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, int(pad_token_id)),
                    next_tokens,
                )
                finished = finished | next_tokens.eq(int(eos_token_id))

            generated = torch.cat([generated, next_tokens[:, None]], dim=1)
            past_key_values = outputs.past_key_values if use_cache else None
            current_input_ids = next_tokens[:, None] if use_cache else generated

            if eos_token_id is not None and bool(finished.all()):
                break

        if return_dict_in_generate:
            return {"sequences": generated}
        return generated

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_key_bias: Optional[Tensor] = None,
        force_hf_generate: bool = False,
        num_beams: int = 1,
        do_sample: bool = False,
        **generate_kwargs: Any,
    ) -> Any:
        """Generate sequences.

        By default, greedy decoding uses the local cached loop. Set
        ``force_hf_generate=True`` or request beams/sampling to use HF generate.
        """
        if not force_hf_generate and int(num_beams) == 1 and not bool(do_sample):
            return self.greedy_generate(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_key_bias=encoder_key_bias,
                num_beams=num_beams,
                do_sample=do_sample,
                **generate_kwargs,
            )

        if BaseModelOutput is None:
            raise ImportError("transformers is required for Hugging Face generation.")
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id).long(),
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            encoder_key_bias=encoder_key_bias,
            num_beams=num_beams,
            do_sample=do_sample,
            **generate_kwargs,
        )
