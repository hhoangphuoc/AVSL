import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
import numpy as np
from modules.av_hubert_layers import (
    LayerNorm,
)

from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextDecoder,
    Speech2TextDecoderLayer,
    Speech2TextAttention,
)
from transformers.activations import ACT2FN

#===============================================================================
#                           AVHuBERTAttention
# ================================================================================
class AVHuBERTAttention(Speech2TextAttention):
    """
    AVHuBERT attention module based on the reference implementation.
    This is a wrapper around Speech2TextAttention but with the possibility
    of different input and output dimensions.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim  # Can be different from embed_dim for cross-attention
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # Projection layers with support for different input/output dimensions
        self.k_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#===============================================================================
#                           AVHuBERTDecoderLayer
# ================================================================================
class AVHuBERTDecoderLayer(Speech2TextDecoderLayer):
    """
    AVHuBERT decoder layer based on the reference implementation.
    Extends Speech2TextDecoderLayer but uses AVHuBERTAttention.
    """
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.d_model

        # Self-attention
        # NOTE: This is by default refer to "eager" attention 
        # implementation @ huggingface/transformers/models/speech_to_text/modeling_speech_to_text.pyhug
        self.self_attn = AVHuBERTAttention(
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function] if hasattr(config, "activation_function") else ACT2FN["gelu"]
        self.activation_dropout = config.activation_dropout if hasattr(config, "activation_dropout") else 0.0

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Cross-attention to encoder outputs
        encoder_hidden_size = getattr(config, "encoder_hidden_size", config.hidden_size)
        self.encoder_attn = AVHuBERTAttention(
            input_dim=encoder_hidden_size,  # This allows different encoder dimensions
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=True,
            is_causal=False,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feed-forward network
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    #------------------------------------------------------------------------------------------------
    # FIXME: REMOVE IF NOT NEEDED   
    # Original forward method from `Speech2TextDecoderLayer`
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     layer_head_mask: Optional[torch.Tensor] = None,
    #     cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = True,
    # ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    #     """
    #     Args:
    #         hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
    #         attention_mask (`torch.FloatTensor`): attention mask of size
    #             `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
    #         encoder_hidden_states (`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
    #         encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
    #             `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
    #         layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
    #             `(encoder_attention_heads,)`.
    #         cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
    #             size `(decoder_attention_heads,)`.
    #         past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
    #         output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all attention layers.
    #     """
    #     residual = hidden_states

    #     # Self-attention block
    #     # Prepare head mask if needed (1.0 = use head, 0.0 = don't use head)
    #     head_mask = layer_head_mask if layer_head_mask is not None else torch.ones(
    #         self.self_attn.num_heads, device=hidden_states.device, dtype=hidden_states.dtype
    #     )
        
    #     # Prepare past key/value for efficient attention
    #     self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
    #     # Apply self-attention
    #     hidden_states, self_attn_weights, present_key_value = self.self_attn(
    #         hidden_states=hidden_states,
    #         past_key_value=self_attn_past_key_value,
    #         attention_mask=attention_mask,
    #         layer_head_mask=head_mask,
    #         output_attentions=output_attentions,
    #     )
    #     hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
    #     hidden_states = residual + hidden_states
    #     hidden_states = self.self_attn_layer_norm(hidden_states)

    #     # Cross-attention block
    #     cross_attn_present_key_value = None
    #     cross_attn_weights = None
        
    #     if encoder_hidden_states is not None:
    #         residual = hidden_states
            
    #         # Cross-attention head mask
    #         cross_attn_head_mask = cross_attn_layer_head_mask if cross_attn_layer_head_mask is not None else torch.ones(
    #             self.encoder_attn.num_heads, device=hidden_states.device, dtype=hidden_states.dtype
    #         )
            
    #         # Prepare key/value from past if available
    #         cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            
    #         # Apply cross-attention
    #         hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
    #             hidden_states=hidden_states,
    #             key_value_states=encoder_hidden_states,
    #             attention_mask=encoder_attention_mask,
    #             layer_head_mask=cross_attn_head_mask,
    #             past_key_value=cross_attn_past_key_value,
    #             output_attentions=output_attentions,
    #         )
    #         hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
    #         hidden_states = residual + hidden_states
    #         hidden_states = self.encoder_attn_layer_norm(hidden_states)
            
    #         # Update present key/value for caching
    #         present_key_value = present_key_value + cross_attn_present_key_value if use_cache else None

    #     # Fully Connected block
    #     residual = hidden_states
    #     hidden_states = self.activation_fn(self.fc1(hidden_states))
    #     hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    #     hidden_states = self.fc2(hidden_states)
    #     hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
    #     hidden_states = residual + hidden_states
    #     hidden_states = self.final_layer_norm(hidden_states)

    #     outputs = (hidden_states,)

    #     if output_attentions:
    #         outputs += (self_attn_weights, cross_attn_weights)

    #     if use_cache:
    #         outputs += (present_key_value,)

    #     return outputs
    #------------------------------------------------------------------------------------------------
#===============================================================================
#                           AVHuBERTDecoder
# ================================================================================
class AVHuBERTDecoder(Speech2TextDecoder):
    """
    AVHuBERT decoder module based on the reference implementation.
    Extends Speech2TextDecoder but uses AVHuBERTDecoderLayer.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Override layers with AVHuBERTDecoderLayer
        self.layers = nn.ModuleList([AVHuBERTDecoderLayer(config) for _ in range(config.decoder_layers)])
    
    #------------------------------------------------------------------------------------------------
    # FIXME: REMOVE IF NOT NEEDED
    #     # Add methods to prepare decoder attention masks if missing
    #     if not hasattr(self, "_prepare_decoder_attention_mask"):
    #         # This method is copied from Speech2TextDecoder
    #         self._prepare_decoder_attention_mask = self._prepare_decoder_attention_mask_method
        
    #     # Add methods to create and expand masks if missing
    #     if not hasattr(self, "_make_causal_mask"):
    #         self._make_causal_mask = self._make_causal_mask_method
            
    #     if not hasattr(self, "_expand_mask"):
    #         self._expand_mask = self._expand_mask_method

    # # Add helper methods that might be missing in some versions
    # def _make_causal_mask_method(self, input_ids_shape, dtype, past_key_values_length=0):
    #     """
    #     Make causal mask used for bi-directional self-attention.
    #     """
    #     bsz, tgt_len = input_ids_shape
    #     mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    #     mask_cond = torch.arange(mask.size(-1))
    #     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    #     mask = mask.to(dtype)

    #     if past_key_values_length > 0:
    #         mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    #     return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # def _expand_mask_method(self, mask, dtype, tgt_len=None):
    #     """
    #     Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    #     """
    #     bsz, src_len = mask.size()
    #     tgt_len = tgt_len if tgt_len is not None else src_len

    #     expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    #     inverted_mask = 1.0 - expanded_mask

    #     return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        
    # def _prepare_decoder_attention_mask_method(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    #     # create causal mask
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     combined_attention_mask = None
    #     if input_shape[-1] > 1:
    #         combined_attention_mask = self._make_causal_mask(
    #             input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
    #         ).to(inputs_embeds.device)

    #     if attention_mask is not None:
    #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #         expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
    #         combined_attention_mask = (
    #             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    #         )

    #     return combined_attention_mask
    #------------------------------------------------------------------------------------------------
