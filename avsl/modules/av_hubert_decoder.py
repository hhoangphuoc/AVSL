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
class AVHuBERTDecoderLayer(nn.Module):
    """
    Decoder layer for the AVHuBERT model.
    
    Args:
        config: Wav2Vec2Config or other compatible config
        decoder_attention_heads (int, optional): Number of attention heads in the self- and cross-attention.
            Defaults to config.decoder_attention_heads.
        decoder_embed_dim (int, optional): Dimensionality of the embedding layer.
            Defaults to config.decoder_embed_dim.
        decoder_ffn_embed_dim (int, optional): Dimensionality of the feed-forward network.
            Defaults to config.decoder_ffn_embed_dim.
        encoder_projection_dim (int, optional): Dimensionality for encoder projection in cross-attention.
            Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to config.dropout.
        attention_dropout (float, optional): Attention dropout rate. Defaults to config.attention_dropout.
        activation_dropout (float, optional): Activation dropout rate. Defaults to config.activation_dropout.
        activation_fn (str, optional): Activation function. Defaults to config.activation_function or "gelu".
        normalize_before (bool, optional): Whether to apply layer norm before attention.
            Defaults to config.decoder_normalize_before.
    """
    
    def __init__(
        self,
        config,
        decoder_attention_heads=None,
        decoder_embed_dim=None,
        decoder_ffn_embed_dim=None,
        encoder_projection_dim=None,
        dropout=None,
        attention_dropout=None,
        activation_dropout=None,
        activation_fn=None,
        normalize_before=None,
    ):
        super().__init__()
        
        # Extract dimensions from config with defaults
        self.embed_dim = decoder_embed_dim or getattr(config, "decoder_embed_dim", 768)
        self.ffn_embed_dim = decoder_ffn_embed_dim or getattr(config, "decoder_ffn_embed_dim", 3072)
        self.attention_heads = decoder_attention_heads or getattr(config, "decoder_attention_heads", 8)
        self.encoder_projection_dim = encoder_projection_dim
        
        # Extract dropout rates from config with defaults
        self.dropout = dropout if dropout is not None else getattr(config, "dropout", 0.1)
        self.attention_dropout = attention_dropout if attention_dropout is not None else getattr(config, "attention_dropout", 0.0)
        self.activation_dropout = activation_dropout if activation_dropout is not None else getattr(config, "activation_dropout", 0.1)
        
        # Get activation function
        self.activation_fn = activation_fn or getattr(config, "activation_function", "gelu")
        if isinstance(self.activation_fn, str):
            self.activation_fn = ACT2FN[self.activation_fn]
            
        # Get normalization setting
        self.normalize_before = normalize_before if normalize_before is not None else getattr(config, "decoder_normalize_before", False)
        
        # Initialize self-attention
        self.self_attn = AVHuBERTAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
        )
        
        # Initialize encoder-decoder attention (cross-attention)
        self.encoder_attn = AVHuBERTAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            input_dim=self.encoder_projection_dim,
            is_cross_attention=True,
        )
        
        # Layer normalization
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feed-forward network
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(self.dropout)
        self.activation_dropout_layer = nn.Dropout(self.activation_dropout)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        """
        Forward pass for a decoder layer.
        
        Args:
            hidden_states (torch.Tensor): Input tensor from previous decoder layer.
            attention_mask (torch.Tensor, optional): Masking tensor for self-attention.
            encoder_hidden_states (torch.Tensor, optional): Output from the encoder.
            encoder_attention_mask (torch.Tensor, optional): Masking tensor for encoder attention.
            layer_head_mask (torch.Tensor, optional): Mask for self-attention heads.
            cross_attn_layer_head_mask (torch.Tensor, optional): Mask for cross-attention heads.
            past_key_value (tuple, optional): Cached key/value states for efficient decoding.
            output_attentions (bool, optional): Whether to return attention weights. Default: False.
            use_cache (bool, optional): Whether to use cached key/values. Default: True.
            
        Returns:
            tuple: Tuple containing:
                - hidden_states (torch.Tensor): Output tensor from this layer.
                - present_key_value (tuple): Updated cached key/value states, if use_cache is True.
                - self_attn_weights (torch.Tensor): Self-attention weights, if output_attentions is True.
                - cross_attn_weights (torch.Tensor): Cross-attention weights, if output_attentions is True.
        """
        
        # Split cached key/values if past_key_value is provided
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None
        
        # Apply pre-normalization if configured (normalize-before)
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
            
        # Self-attention block
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        
        # Apply dropout and residual connection
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states
        
        # Apply post-normalization if not using pre-normalization
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
            
        # Cross-attention block (encoder-decoder attention)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # Apply pre-normalization for cross-attention if configured
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)
                
            # Apply cross-attention
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            
            # Apply dropout and residual connection
            hidden_states = self.dropout_layer(hidden_states)
            hidden_states = residual + hidden_states
            
            # Apply post-normalization if not using pre-normalization
            if not self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)
                
            # Update present_key_value with cross-attention key/values
            if use_cache:
                present_key_value = present_key_value + cross_attn_present_key_value
                
        # Feed-forward network block
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
            
        # Apply feed-forward network
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        # Apply dropout and residual connection
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states
        
        # Apply post-normalization if not using pre-normalization
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
            
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (present_key_value,)
            
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
            
        return outputs

#===============================================================================
#                           AVHuBERTDecoder
# ================================================================================
class AVHuBERTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a `AVHuBERTDecoderLayer`.
    
    Args:
        config: Wav2Vec2Config or other compatible config
        vocab_size (int, optional): Size of the vocabulary for embeddings. Defaults to config.vocab_size.
        decoder_embed_dim (int, optional): Dimensionality of the embedding layer. Defaults to config.decoder_embed_dim.
        decoder_ffn_embed_dim (int, optional): Dimensionality of the feed-forward network. Defaults to config.decoder_ffn_embed_dim.
        decoder_layers (int, optional): Number of decoder layers. Defaults to config.decoder_layers.
        decoder_attention_heads (int, optional): Number of attention heads. Defaults to config.decoder_attention_heads.
        dropout (float, optional): Dropout rate. Defaults to config.dropout.
        attention_dropout (float, optional): Attention dropout rate. Defaults to config.attention_dropout.
        activation_dropout (float, optional): Activation dropout rate. Defaults to config.activation_dropout.
        layerdrop (float, optional): Layer dropout rate. Defaults to config.decoder_layerdrop.
        max_target_positions (int, optional): Maximum sequence length. Defaults to 1024.
        normalize_before (bool, optional): Whether to apply layer norm before attention. Defaults to config.decoder_normalize_before.
        scale_embedding (bool, optional): Whether to scale embeddings. Defaults to not config.no_scale_embedding.
        embed_scale (float, optional): Embedding scale factor. Defaults to sqrt(decoder_embed_dim) if scale_embedding is True.
        share_decoder_input_output_embed (bool, optional): Whether to share input/output embeddings. Defaults to config.share_decoder_input_output_embed.
    """
    def __init__(
        self,
        config,
        vocab_size=None,
        decoder_embed_dim=None,
        decoder_ffn_embed_dim=None,
        decoder_layers=None,
        decoder_attention_heads=None,
        dropout=None,
        attention_dropout=None,
        activation_dropout=None,
        layerdrop=None,
        max_target_positions=1024,
        normalize_before=None,
        scale_embedding=None,
        embed_scale=None,
        share_decoder_input_output_embed=None,
    ):
        super().__init__()
        
        # Extract parameters from config with defaults
        self.vocab_size = vocab_size or getattr(config, "vocab_size", 10000)
        self.embed_dim = decoder_embed_dim or getattr(config, "decoder_embed_dim", 768)
        self.ffn_embed_dim = decoder_ffn_embed_dim or getattr(config, "decoder_ffn_embed_dim", 3072)
        self.num_layers = decoder_layers or getattr(config, "decoder_layers", 6)
        self.num_attention_heads = decoder_attention_heads or getattr(config, "decoder_attention_heads", 8)
        self.dropout = dropout if dropout is not None else getattr(config, "dropout", 0.1)
        self.attention_dropout = attention_dropout if attention_dropout is not None else getattr(config, "attention_dropout", 0.0)
        self.activation_dropout = activation_dropout if activation_dropout is not None else getattr(config, "activation_dropout", 0.1)
        self.layerdrop = layerdrop if layerdrop is not None else getattr(config, "decoder_layerdrop", 0.0)
        self.max_target_positions = max_target_positions
        self.normalize_before = normalize_before if normalize_before is not None else getattr(config, "decoder_normalize_before", False)
        self.scale_embedding = scale_embedding if scale_embedding is not None else not getattr(config, "no_scale_embedding", False)
        
        # Get position embedding type and learned position settings
        self.learned_pos = getattr(config, "decoder_learned_pos", False)
        
        # Initialize embedding scale
        if self.scale_embedding and embed_scale is None:
            self.embed_scale = math.sqrt(self.embed_dim)
        else:
            self.embed_scale = embed_scale or 1.0
            
        # Initialize embedding layer
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # Initialize position embeddings
        if self.learned_pos:
            self.embed_positions = LearnedPositionalEmbedding(
                self.max_target_positions, 
                self.embed_dim
            )
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.max_target_positions, 
                self.embed_dim
            )
            
        # Initialize decoder layers
        self.layers = nn.ModuleList([
            AVHuBERTDecoderLayer(
                config,
                decoder_attention_heads=self.num_attention_heads,
                decoder_embed_dim=self.embed_dim,
                decoder_ffn_embed_dim=self.ffn_embed_dim,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                normalize_before=self.normalize_before,
            )
            for _ in range(self.num_layers)
        ])
        
        # Add layer normalization if needed
        if self.normalize_before:
            self.layer_norm = nn.LayerNorm(self.embed_dim)
            
        # Initialize projection to vocabulary
        self.share_decoder_input_output_embed = share_decoder_input_output_embed if share_decoder_input_output_embed is not None else getattr(config, "share_decoder_input_output_embed", False)
        
        if self.share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5)
            
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        combined_attention_mask = None
        if input_shape[1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
        
    def _make_causal_mask(self, input_ids_shape, dtype, past_key_values_length=0):
        """
        Make causal mask used for self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")), dtype=dtype)
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
            
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Extract settings with defaults
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else True
        return_dict = return_dict if return_dict is not None else True
        
        # Check if inputs_embeds was provided
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        # Past key values length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # Apply embedding scale if configured
        if self.scale_embedding:
            inputs_embeds = inputs_embeds * self.embed_scale
            
        # Get position embeddings and add to input
        if past_key_values_length > 0 and self.learned_pos:
            # If using learned position embeddings with caching, adjust positions
            position_ids = torch.arange(
                past_key_values_length, input_shape[1] + past_key_values_length, 
                dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)
            position_embeddings = self.embed_positions(position_ids)
        else:
            # Otherwise use regular position embeddings
            position_embeddings = self.embed_positions(inputs_embeds, past_key_values_length)
            
        # Combine token and position embeddings
        hidden_states = inputs_embeds + position_embeddings
            
        # Prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask=None,  # We don't use attention_mask for decoder-only autoregressive tasks
            input_shape=input_shape,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # Prepare encoder attention mask if encoder_hidden_states is provided
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                batch_size, seq_length = encoder_hidden_states.shape[:2]
                encoder_attention_mask = torch.ones(batch_size, seq_length, device=encoder_hidden_states.device)
                
            # Convert encoder_attention_mask to correct format for cross-attention
            encoder_attention_mask = self._expand_mask(encoder_attention_mask, hidden_states.dtype, tgt_len=input_shape[1])
        
        # Initialize lists for collecting outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        
        # Apply layerdrop to randomly drop certain layers during training
        active_layers = list(range(len(self.layers)))
        if self.training and self.layerdrop > 0:
            active_layers = [layer_idx for layer_idx in active_layers if random.random() > self.layerdrop]
            
        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Skip non-active layers if using layerdrop
            if self.training and idx not in active_layers:
                continue
                
            # Collect hidden states for output_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Get past key values for current layer
            layer_past = past_key_values[idx] if past_key_values is not None else None
            
            # Process through the decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            # Unpack layer outputs
            hidden_states = layer_outputs[0]
            
            # Collect attention weights if requested
            if output_attentions:
                all_self_attns += (layer_outputs[2 if use_cache else 1],)
                
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[3 if use_cache else 2],)
                    
            # Collect cache values if requested
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
                
        # Apply final layer norm if present (pre-normalization)
        if self.normalize_before and hasattr(self, "layer_norm"):
            hidden_states = self.layer_norm(hidden_states)
            
        # Collect final hidden states if requested
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        # Project hidden states to vocabulary space
        logits = self.output_projection(hidden_states)
        
        # Prepare next cache if using caching
        next_cache = next_decoder_cache if use_cache else None
        
        # Return appropriate output format based on return_dict
        if not return_dict:
            return tuple(v for v in [
                logits, 
                next_cache, 
                all_hidden_states, 
                all_self_attns, 
                all_cross_attentions
            ] if v is not None)
            
        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

#===============================================================================
#                  AVHuBERTSinusoidalPositionalEmbedding
# ================================================================================
class AVHuBERTSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self._get_embedding(num_positions, embedding_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def _get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input_shape, past_key_values_length=0):
        """Input is expected to be of shape [bsz x seqlen]."""
        bsz, seq_len = input_shape
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, device=self._float_tensor.device
        ).expand(bsz, -1)

        return self._get_embedding(
            positions.size(-1) + self.padding_idx + 1 if self.padding_idx is not None else positions.size(-1),
            self.embedding_dim,
            self.padding_idx,
        )[positions.long()].detach()
