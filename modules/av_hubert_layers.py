import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, Tuple, Union, Dict, List

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

from modules.resnet import ResNetEncoderLayer


#==============================================================================
#                           AudioEncoder
# ================================================================================
class AudioEncoderLayer(nn.Module):
    """
    Audio encoder component of AV-HuBERT.
    Processes audio input using convolutional feature extraction
    followed by transformer encoder layers.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        feat_extract_dropout: float = 0.0,
        feat_extract_activation: str = "gelu",
        conv_dim: List[int] = [512, 512, 512, 512, 512, 512, 512],
        conv_kernel: List[int] = [10, 3, 3, 3, 3, 2, 2],
        conv_stride: List[int] = [5, 2, 2, 2, 2, 2, 2],
        use_position_embeddings: bool = True,
        position_embedding_type: str = "conv", # "learned" or "conv"
        layerdrop: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layerdrop = layerdrop
        
        # Feature extraction for audio
        if len(conv_dim) != len(conv_kernel) or len(conv_dim) != len(conv_stride):
            raise ValueError("Configuration lists for audio convolutional layers must have the same length")
        
        conv_layers = list(zip(conv_dim, conv_kernel, conv_stride))
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=feat_extract_dropout,
        )
        
        # Feature projection layer
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feature_projection = nn.Linear(conv_dim[-1], hidden_size)
        self.feature_dropout = nn.Dropout(hidden_dropout_prob)
        
        # Positional embedding
        self.position_embeddings = None
        if use_position_embeddings:
            if position_embedding_type == "conv":
                self.position_embeddings = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=128,
                    padding=128 // 2,
                    groups=16,
                )
            else:  # "learned"
                # Will be initialized with proper sequence length during forward pass
                self.position_embeddings = None
                self.register_buffer("position_ids", torch.arange(2048).expand((1, -1)))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        
        if input_values is None:
            raise ValueError("You have to specify input_values")
        
        # Feature extraction
        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, C)
        
        # Apply feature projection
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.feature_dropout(hidden_states)
        
        # Apply positional embeddings if available
        if isinstance(self.position_embeddings, nn.Conv1d):
            # Convolutional positional embeddings
            position_embeddings = self.position_embeddings(hidden_states.transpose(1, 2))
            position_embeddings = position_embeddings.transpose(1, 2)
            hidden_states = hidden_states + position_embeddings
        
        # Apply Layer Normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Generate attention_mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values)
        
        # Convert attention_mask to the format expected by the transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Collect hidden states if output_hidden_states is True
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # Apply encoder layers
        for i, layer in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Apply layer drop if necessary during training
            dropout_probability = np.random.random()
            if self.training and (dropout_probability < self.layerdrop):
                continue
            
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
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
#                           VisualEncoder
# ================================================================================
class VisualEncoderLayer(nn.Module):
    """
    Visual Encoder for AV-HuBERT, including visual frontend (ResNet)
    and Transformer Encoder layers for sequence modeling.
    """
    def __init__(
        self,
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layerdrop = layerdrop
        
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
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=hidden_dropout_prob,
                attention_dropout=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
                activation="gelu",
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Initialize other weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights for non-pretrained modules."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Skip initialization for pretrained modules
            if not isinstance(module, nn.Conv2d) or not hasattr(module, '_is_pretrained'):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
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
        hidden_states = self.visual_frontend(input_values)  # (B, C, T)
        
        # Transpose to (B, T, C) for projection
        hidden_states = hidden_states.transpose(1, 2)
        
        # Project features to transformer dimension
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.feature_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_values.size(0), hidden_states.size(1)), 
                device=input_values.device
            )
        
        # Convert attention_mask to the format expected by the transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Collect outputs if requested
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # Apply transformer encoder layers
        for i, layer in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Apply layerdrop if needed
            dropout_probability = torch.rand(1).item()
            if self.training and (dropout_probability < self.layerdrop):
                continue
            
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
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
#                                HELPER LAYERS
# ================================================================================


#------------------------------------------------------------------------------
#                           LayerNorm
#------------------------------------------------------------------------------
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

#------------------------------------------------------------------------------
#                           ConvFeatureExtractionModel
#------------------------------------------------------------------------------
class ConvFeatureExtractionModel(nn.Module):
    """
    Convolutional feature extraction model for audio signals.
    Based on the AV-HuBERT implementation but using PyTorch directly.
    """
    def __init__(self, conv_layers: List[Tuple[int, int, int]], dropout: float = 0.0):
        super().__init__()
        
        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.GroupNorm(1, n_out),
                nn.GELU(),
            )
        
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "conv_layers should be a list of tuples (dim, kernel_size, stride)"
            (dim, k, stride) = cl
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T)
        x = x.unsqueeze(1)  # (B, 1, T)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        return self.dropout(x)

#------------------------------------------------------------------------------
#                           TransformerEncoderLayer
#------------------------------------------------------------------------------
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