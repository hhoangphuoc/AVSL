# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ORIGINAL SOURCE CODE: https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, Tuple, Union, Dict, List

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderLayer,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2Encoder,
    Wav2Vec2FeedForward,
    WAV2VEC2_ATTENTION_CLASSES,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection
)

from modules.resnet import ResNetEncoderLayer #RESNET ENCODER FOR VISUAL ENCODING


#==============================================================================
#                  AVWav2Vec2PositionalConvEmbedding
# ================================================================================
class AVWav2Vec2PositionalConvEmbedding(Wav2Vec2PositionalConvEmbedding):
    """
    Custom positional embedding for AV-HuBERT, extends Wav2Vec2PositionalConvEmbedding
    """
    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)

#==============================================================================
#                           AVHubertEncoderLayer
# ================================================================================
class AVHubertBaseEncoderLayer(Wav2Vec2EncoderLayer):
    """
    Custom encoder layer for AV-HuBERT, extends Wav2Vec2EncoderLayer
    """
    def __init__(self, config):
        super().__init__(config)
        self.attention = WAV2VEC2_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = AVWav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # hidden_states = self.layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

#==============================================================================
#                           AVWav2Vec2FeedForward
# ================================================================================
class AVWav2Vec2FeedForward(Wav2Vec2FeedForward):
    """
    Custom feed-forward for AV-HuBERT, extends Wav2Vec2FeedForward
    """
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = nn.functional.gelu if config.hidden_act == "gelu" else nn.functional.relu
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

#==============================================================================
#                           AVHubertEncoder
# ================================================================================
class AVHubertBaseEncoder(Wav2Vec2Encoder):
    """
    Audio and Visual encoder component of AV-HuBERT.
    Extends the Hugging Face implementation of Wav2Vec2Encoder but with custom layers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([AVHubertBaseEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.pos_conv_embed = AVWav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

#==============================================================================
#                           AudioEncoderLayer
# ================================================================================
class AudioEncoderLayer(AVHubertBaseEncoder):
    """
    Audio encoder component of AV-HuBERT.
    Uses AVHubertEncoder for consistency with the original implementation,
    but also incorporates Wav2Vec2Model's feature extraction and projection.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the Wav2Vec2 feature extractor and projection
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        
        # Gradient multiply factor for feature extractor
        self.feature_grad_mult = getattr(config, "feature_grad_mult", 1.0)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through the audio encoder.
        
        Args:
            input_values: Audio input of shape (batch_size, sequence_length)
            attention_mask: Optional mask for padding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state (batch_size, sequence_length, hidden_size)
                - hidden_states (optional)
                - attentions (optional)
        """
        # Extract features
        if input_values is None:
            raise ValueError("You have to specify input_values")
            
        # Get features from feature extractor
        extract_features = self.feature_extractor(input_values)
        
        # Apply gradient multiplication if needed
        if self.feature_grad_mult != 1.0 and self.training:
            extract_features = GradMultiply.apply(extract_features, self.feature_grad_mult)
        
        # Apply feature projection
        hidden_states, projected_features = self.feature_projection(extract_features)
        
        # Forward through AVHubertEncoder
        encoder_outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return encoder_outputs
    
    def extract_features(self, input_values, attention_mask=None):
        """
        Extract features from raw audio input using feature extractor only.
        """
        # Extract features
        extract_features = self.feature_extractor(input_values)
        
        # Apply gradient multiplication if needed
        if self.feature_grad_mult != 1.0 and self.training:
            extract_features = GradMultiply.apply(extract_features, self.feature_grad_mult)
        
        # Apply feature projection
        hidden_states, _ = self.feature_projection(extract_features)
        
        return hidden_states

#==============================================================================
#                           VisualEncoder
# ================================================================================
class VisualEncoderLayer(AVHubertBaseEncoder):
    """
    Visual Encoder for AV-HuBERT, including visual frontend (ResNet)
    and transformer encoder layers (AVHubertEncoder).
    """
    def __init__(
        self,
        config,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        frontend_channels: int = 64,
        backbone_channels: int = 512,
        use_pretrained_frontend: bool = False,
        frontend_weights_path: Optional[str] = None,
        relu_type: str = 'prelu',
        layerdrop: float = 0.0,
        feature_grad_mult: float = 1.0,
    ):
        #-------------------------------------------------------------------------------------------------------------------
        # Update config with visual parameters if needed
        if not hasattr(config, "hidden_size") or config.hidden_size != hidden_size:
            config.hidden_size = hidden_size
        if not hasattr(config, "num_hidden_layers") or config.num_hidden_layers != num_hidden_layers:
            config.num_hidden_layers = num_hidden_layers
        if not hasattr(config, "num_attention_heads") or config.num_attention_heads != num_attention_heads:
            config.num_attention_heads = num_attention_heads
        if not hasattr(config, "intermediate_size") or config.intermediate_size != intermediate_size:
            config.intermediate_size = intermediate_size
        if not hasattr(config, "hidden_dropout_prob") or config.hidden_dropout_prob != hidden_dropout_prob:
            config.hidden_dropout_prob = hidden_dropout_prob
        if not hasattr(config, "attention_probs_dropout_prob") or config.attention_probs_dropout_prob != attention_probs_dropout_prob:
            config.attention_probs_dropout_prob = attention_probs_dropout_prob
        if not hasattr(config, "layer_norm_eps") or config.layer_norm_eps != layer_norm_eps:
            config.layer_norm_eps = layer_norm_eps
        if not hasattr(config, "layerdrop") or config.layerdrop != layerdrop:
            config.layerdrop = layerdrop
        #------------------------------------------------------------------------------------------------------------------- 

            
        # Initialize the base AVHubertEncoder
        super().__init__(config)
        
        self.hidden_size = hidden_size
        self.feature_grad_mult = feature_grad_mult
        
        # Visual frontend
        self.visual_frontend = ResNetEncoderLayer(
            frontend_out_channels=frontend_channels,
            backbone_out_channels=backbone_channels,
            relu_type=relu_type,
            weights=frontend_weights_path if use_pretrained_frontend else None
        )
        
        # Projection layer to match transformer dimension
        self.feature_projection = nn.Linear(backbone_channels, hidden_size)
        self.feature_dropout = nn.Dropout(hidden_dropout_prob)
        self.feature_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Initialize frontend and projection weights
        self._init_frontend_weights()
    
    def _init_frontend_weights(self):
        """Initialize the weights for non-pretrained modules."""
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Skip initialization for pretrained modules
                if not isinstance(module, nn.Conv2d) or not hasattr(module, '_is_pretrained'):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
        # Initialize only the projection layer
        self.feature_projection.apply(_init_weights)
        self.feature_layer_norm.apply(_init_weights)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,  # (B, C, T, H, W)
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through the visual encoder.
        
        Args:
            input_values: Video frames tensor of shape (B, C, T, H, W)
            attention_mask: Optional mask for padding in the time dimension (B, T)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state (B, T, hidden_size)
                - hidden_states (optional)
                - attentions (optional)
        """
        if input_values is None:
            raise ValueError("You have to specify input_values")
        
        # Extract features using the visual frontend (ResNet)
        features = self.visual_frontend(input_values)  # (B, C, T)
        
        # Apply gradient multiplication if needed
        if self.feature_grad_mult != 1.0 and self.training:
            features = GradMultiply.apply(features, self.feature_grad_mult)
        
        # Transpose to (B, T, C) for projection
        features = features.transpose(1, 2)
        
        # Project features to transformer dimension
        hidden_states = self.feature_projection(features)
        hidden_states = self.feature_dropout(hidden_states)
        hidden_states = self.feature_layer_norm(hidden_states)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_values.size(0), hidden_states.size(1)), 
                device=input_values.device
            )
        
        # Process through AVHubertEncoder
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def extract_features(self, input_values, attention_mask=None):
        """
        Extract features from the visual frontend
        """
        # Extract features using the visual frontend (ResNet)
        features = self.visual_frontend(input_values)  # (B, C, T)
        
        # Apply gradient multiplication if needed
        if self.feature_grad_mult != 1.0 and self.training:
            features = GradMultiply.apply(features, self.feature_grad_mult)
            
        # Convert to (B, T, C) format
        features = features.transpose(1, 2)
        
        return features
    
#--------------------------------------------------------------------------------------------------------
# LayerNorm implementation consistent with Hugging Face transformers
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

# Gradient multiply operation
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


#----------------------------------------------------------------------------------------------------------
#                                           TransformerEncoderLayer
#---------------------------------------------------------------------------------------------------------- 
class TransformerEncoderLayer(nn.Module):
    """
    Custom implementation of a transformer encoder layer for the visual encoder.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=True)
        
        # Implementation of feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == "gelu" else F.relu
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            hidden_states: Input tensor of shape (B, T, d_model)
            attention_mask: Attention mask of shape (B, 1, 1, T)
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple containing:
                - Output tensor of shape (B, T, d_model)
                - Optional attention weights if output_attentions is True
        """
        # Self-attention block
        residual = hidden_states
        
        # Apply attention mask
        if attention_mask is not None:
            # Reshape mask for the multihead attention expected format
            attn_mask = attention_mask.squeeze(1).squeeze(1)
        else:
            attn_mask = None
        
        # Multi-head attention
        hidden_states, attn_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=attn_mask,
            need_weights=output_attentions,
        )
        
        hidden_states = self.dropout1(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Feedforward block
        residual = hidden_states
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs 
#----------------------------------------------------------------------------------------------------------
