import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import Tensor

from transformers import Cache, HybridCache, StaticCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers import (
    PreTrainedModel,
    AutoConfig,
    PaliGemmaPreTrainedModel,
    AutoModelForCausalLM,
    GenerationMixin,
)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig
from transformers.models.donut.modeling_donut_swin import DonutSwinModel
from transformers.utils import logging


from .configuration_divedoc import SwinPamVisionEncoderConfig, DIVEdocConfig
from typing import List, Optional, Tuple, Union, Literal
from dataclasses import dataclass


logger = logging.get_logger(__name__)



class PAM(nn.Module):
    def __init__(
        self,
        sequence_mapping_layer_type: Literal[
            "linear_projection", "bilinear", "bicubic", "nearest-exact"
        ] = "bilinear",
        student_fmap_dim: Tuple[int, int] = (80, 60),
        student_embedding_dim: int = 1024,
        teacher_fmap_dim: Tuple[int, int] = (64, 64),
        teacher_embedding_dim: int = 1152,
    ):
        super().__init__()
        self.sequence_mapping_layer_type = sequence_mapping_layer_type
        self.sequence_mapping_layer = (
            nn.Linear(
                student_fmap_dim[0] * student_fmap_dim[1],
                teacher_fmap_dim[0] * teacher_fmap_dim[1],
            )
            if sequence_mapping_layer_type == "linear_projection"
            else None
        )
        self.embedding_projection_layer = nn.Sequential(
            nn.Linear(student_embedding_dim, teacher_embedding_dim),
            nn.LayerNorm((teacher_embedding_dim,), eps=1e-06),
        )

        self.student_fmap_dim = student_fmap_dim
        self.student_embedding_dim = student_embedding_dim
        self.teacher_fmap_dim = teacher_fmap_dim
        self.teacher_embedding_dim = teacher_embedding_dim

        print(self.student_fmap_dim)

    # take input x of shape (Batch, Nb_token, Dim_embedding)
    def forward(self, x: Tensor) -> Tensor:
        #
        """
        if x.shape[1] != self.student_fmap_dim[0] * self.student_fmap_dim[1] or x.shape[2] != self.student_embedding_dim:
            raise ValueError(f"Expected input shape (*, {self.student_fmap_dim[0] * self.student_fmap_dim[1],self.student_embedding_dim}), "
                             f"but got {x.shape}")
        """

        if x.shape[1] != (self.teacher_fmap_dim[0] * self.teacher_fmap_dim[1]):
            print(x.shape[1])
            print(self.teacher_fmap_dim[0] * self.teacher_fmap_dim[1])
            print("Resizing")
            if self.sequence_mapping_layer_type == "linear_projection":
                x = torch.permute(x, (0, 2, 1))
                x = self.sequence_mapping_layer(x)
                x = torch.permute(x, (0, 2, 1))

            elif self.sequence_mapping_layer_type in [
                "bilinear",
                "bicubic",
                "nearest-exact",
            ]:
                batch_size, _, embedding_size = x.size()
                x = x.view(
                    batch_size,
                    self.student_fmap_dim[0],
                    self.student_fmap_dim[1],
                    embedding_size,
                ).permute(0, 3, 1, 2)
                x = F.interpolate(
                    x, size=self.teacher_fmap_dim, mode=self.sequence_mapping_layer_type
                )  # Shape: (1, D, target_height, target_width)
                x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, embedding_size)

            else:
                raise ValueError(
                    "Need a sequence mapping type in [linear_projection, bilinear",
                    "bicubic",
                    "nearest-exact], got {sequence_mapping_layer_type}",
                )

        x = self.embedding_projection_layer(x)
        print(x.shape)
        return x


class SwinPam(nn.Module):
    def __init__(
        self,
        encoder_config: AutoConfig,
        pam_sequence_mapping_layer_type: Literal[
            "linear_projection", "bilinear", "bicubic", "nearest-exact"
        ] = "bilinear",
        pam_student_fmap_dim: Tuple[int, int] = (80, 60),
        pam_student_embedding_dim: int = 1024,
        pam_teacher_fmap_dim: Tuple[int, int] = (64, 64),
        pam_teacher_embedding_dim: int = 1152,
    ):
        super().__init__()
        self.encoder_model = DonutSwinModel(encoder_config)
        print(pam_student_fmap_dim)
        self.pam = PAM(
            sequence_mapping_layer_type=pam_sequence_mapping_layer_type,
            student_fmap_dim=pam_student_fmap_dim,
            student_embedding_dim=pam_student_embedding_dim,
            teacher_fmap_dim=pam_teacher_fmap_dim,
            teacher_embedding_dim=pam_teacher_embedding_dim,
        )

    def forward(self, x):
        x = self.encoder_model(x).last_hidden_state
        x = self.pam(x)
        return x


@dataclass
class SwinPamVisionEncoderOutput(ModelOutput):
    """
    Base class for PaliGemmacausal language model (or autoregressive) outputs.
    Args:
        last_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    last_hidden_states: Optional[torch.FloatTensor] = None


class SwinPamVisionEncoder(PreTrainedModel):
    config_class = SwinPamVisionEncoderConfig
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SwinPam(
            config.encoder_config,
            config.pam_config.sequence_mapping_layer_type,
            config.pam_config.student_fmap_dim,
            config.pam_config.student_embedding_dim,
            config.pam_config.teacher_fmap_dim,
            config.pam_config.teacher_embedding_dim,
        )

    def forward(self, x):
        x = self.model(x)
        return BaseModelOutput(last_hidden_state=x)


# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/modeling_paligemma.py
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.pam_config.teacher_embedding_dim,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


# Copied & Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/modeling_paligemma.py
_CONFIG_FOR_DOC = "DIVEdocConfig"


class DIVEdoc(PaliGemmaPreTrainedModel, GenerationMixin):
    config_class = DIVEdocConfig

    def __init__(self, config: DIVEdocConfig):
        super().__init__(config)

        print(f"Vision config in end-to-end model: {config.vision_config.model_type}")
        if config.vision_config.model_type == "swinpam":
            self.vision_tower = SwinPamVisionEncoder(config=config.vision_config)

        else:
            raise ValueError("Unknown model_type in vision_config")

        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModelForCausalLM.from_config(config=config.text_config)

        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_dtype(self):
        return self.dtype

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
        is_training: bool = None,
        dtype=None,  # to handle quantized finetuning issue when model switch between 4 or 8bit and float
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)

        # Handle the case when the model is quantized in 4 or 8 bit

        if dtype is not None:
            min_dtype = torch.finfo(dtype).min
        else:
            min_dtype = torch.finfo(self.get_dtype()).min

        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask
        """ initial line but changed for quantization processing
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        """
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=cache_position.device,
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # First unmask prefix tokens during training
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)

        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.
        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @replace_return_docstrings(
        output_type=PaliGemmaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, PaliGemmaCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
        Returns:
        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        # save the original dtype before switching to 4bit when quantization
        dtype = self.get_dtype()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = (
                cache_position.unsqueeze(0) + 1
            )  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_index,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
            else:
                special_image_mask = (
                    input_ids == self.config.image_token_index
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )

            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(
                input_ids == self.pad_token_id, self.config.ignore_index, labels
            )

        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            inputs_embeds,
            is_training,
            dtype=dtype,
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(
                    logits.device
                )
                shift_logits = shift_logits[
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = shift_labels[
                    shift_attention_mask.to(shift_labels.device) != 0
                ].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)

            valid_mask = flat_labels != -100

            flat_labels = flat_labels[valid_mask]
            flat_logits = flat_logits[valid_mask]

            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                input_tensor,
                is_training,
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs


def get_model():
    model = DIVEdoc.from_pretrained(
        "JayRay5/DIVE-Doc-FRD", trust_remote_code=True, revision="37d4cc1859b1cb6691930fe573e33b0a88928c59"
    ).eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
