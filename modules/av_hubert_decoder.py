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
        self.input_dim = input_dim if input_dim is not None else embed_dim
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

        # Override projections to handle different input and output dimensions
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

        # Override attention modules to use AVHuBERTAttention
        # NOTE: This is by default refer to "eager" attention implementation @ huggingface/transformers/models/speech_to_text/modeling_speech_to_text.pyhug
        self.self_attn = AVHuBERTAttention(
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = AVHuBERTAttention(
            config.encoder_hidden_size,  # This allows different encoder dimensions
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=False,
            config=config,
        )

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

#===============================================================================