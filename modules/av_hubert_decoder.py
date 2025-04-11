import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
from modules.av_hubert_layers import (
    LayerNorm,
)
    
#===============================================================================
#                           AVHuBERTDecoder
# ================================================================================
class AVHuBERTDecoder(nn.Module):
    """
    Transformer decoder for AV-HuBERT model.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        embed_scale: float = None,
        layerdrop: float = 0.0,
        use_learned_position_embeddings: bool = True,
        share_embedding_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.layerdrop = layerdrop
        self.padding_idx = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        
        # Initialize token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Use sqrt(hidden_size) as scale if not provided
        self.embed_scale = math.sqrt(hidden_size) if embed_scale is None else embed_scale
        
        # Initialize positional embeddings
        if use_learned_position_embeddings:
            self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)
            # Create position IDs tensor (0, 1, 2, ...)
            self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        else:
            self.embed_positions = None
            
        # Optional layer normalization after embeddings
        self.layernorm_embedding = LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            AVHuBERTDecoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Initialize outputs
        if share_embedding_weights:
            # Tie input and output embeddings
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the decoder."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length=0):
        """
        Create causal mask and combine with attention mask if provided.
        """
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            ).to(attention_mask.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass for the decoder.
        
        Args:
            input_ids: Decoder input token IDs, shape (B, T)
            attention_mask: Attention mask for input_ids, shape (B, T)
            encoder_hidden_states: Hidden states from encoder for cross-attention
            encoder_attention_mask: Attention mask for encoder_hidden_states
            past_key_values: Cached past key values for faster inference
            inputs_embeds: Pre-computed embeddings instead of input_ids
            use_cache: Whether to return past key values for faster inference
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state: (B, T, hidden_size)
                - past_key_values: Cache for faster inference (optional)
                - hidden_states: All hidden states (optional)
                - attentions: Attention weights (optional)
                - cross_attentions: Cross-attention weights (optional)
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Get inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
            
        # Calculate past length if using past key values
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            
        # Embed tokens if not already embedded
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            
        # Apply positional embeddings if available
        if self.embed_positions is not None:
            # Get positional embeddings
            positions = self.position_ids[:, past_key_values_length:past_key_values_length + input_shape[1]]
            position_embeddings = self.embed_positions(positions)
            hidden_states = inputs_embeds + position_embeddings
        else:
            hidden_states = inputs_embeds
            
        # Apply layer normalization if available
        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)
            
        # Apply dropout
        hidden_states = self.embedding_dropout(hidden_states)
            
        # Prepare attention masks
        # This depends on the specific transformer implementation, so we'll use a simplified version
        # Assume attention_mask is (B, T) with 1s for tokens to attend and 0s for padding
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Convert masks to the format expected by nn.MultiheadAttention
        # Convert (B, T) -> (B, T) where True means token is padding
        padding_mask = attention_mask.eq(0)
        
        # Prepare encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # Convert (B, S) -> (B, S) where True means token is padding
            encoder_padding_mask = encoder_attention_mask.eq(0)
        else:
            encoder_padding_mask = None
            
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = None
            
        # Apply decoder layers with optional layerdrop
        for idx, decoder_layer in enumerate(self.layers):
            # Add hidden states if needed
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Apply layerdrop if in training
            dropout_probability = np.random.random()
            if self.training and (dropout_probability < self.layerdrop):
                continue
                
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=padding_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_padding_mask,
                need_weights=output_attentions,
            )
                
            hidden_states = layer_outputs[0]
                
            # Add attention weights if needed
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                    
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
                    
        # Add final hidden states if needed
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        # For interface compatibility
        if not return_dict:
            return tuple(v for v in [
                hidden_states, 
                next_decoder_cache, 
                all_hidden_states, 
                all_self_attentions, 
                all_cross_attentions
            ] if v is not None)
            
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
    def output_layer(self, features):
        """Project features to vocabulary."""
        if self.output_projection is None:
            # Tie weights between input embeddings and output projection
            return F.linear(features, self.embed_tokens.weight)
        else:
            return self.output_projection(features)

#===============================================================================
#                           AVHuBERTDecoderLayer
# ================================================================================

class AVHuBERTDecoderLayer(nn.Module):
    """
    Transformer decoder layer used in the AV-HuBERT decoder.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        is_decoder: bool = True,
        add_cross_attention: bool = True,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=True
        )
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.self_attn_layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        
        # Cross-attention
        self.cross_attention = add_cross_attention
        if add_cross_attention:
            self.encoder_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=attention_dropout, batch_first=True
            )
            self.encoder_attn_layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        
        # Feed-forward network
        self.activation_fn = F.gelu if activation == "gelu" else F.relu
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.final_layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for a decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (B, T, d_model)
            attention_mask: Self-attention mask, usually causal
            encoder_hidden_states: Encoder outputs for cross-attention
            encoder_attention_mask: Mask for encoder outputs
            need_weights: Whether to return attention weights
            layer_head_mask: Mask applied to attention heads in self-attention
            cross_attn_layer_head_mask: Mask applied to attention heads in cross-attention
            
        Returns:
            Tuple containing:
                - Output tensor of shape (B, T, d_model)
                - Self-attention weights (optional)
                - Cross-attention weights (optional)
        """
        residual = hidden_states
        
        # Self-attention
        hidden_states, self_attn_weights, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=attention_mask if attention_mask is not None else None,
            need_weights=need_weights,
            attn_mask=None,  # Causal mask is handled separately with key_padding_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Cross-attention
        cross_attn_weights = None
        if self.cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights, _ = self.encoder_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=encoder_attention_mask if encoder_attention_mask is not None else None,
                need_weights=need_weights,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        # For interface compatibility
        outputs = (hidden_states,)
        if need_weights:
            outputs += (self_attn_weights, cross_attn_weights)
        else:
            outputs += (None, None)
            
        return outputs

#===============================================================================