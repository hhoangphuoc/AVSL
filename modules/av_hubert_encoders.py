import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple, Union, Dict, List

from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
from modules.av_hubert_layers import (
    AudioEncoderLayer, 
    VisualEncoderLayer,
    LayerNorm,
)


# ================================================================================
#                           AVHuBERTEncoderWrapper
# ================================================================================
class AVHuBERTEncoderWrapper(nn.Module):
    """
    A wrapper module that manages both audio and visual encoders for AV-HuBERT.
    This module can process audio-only, visual-only, or audio-visual inputs 
    based on the configuration.
    """
    def __init__(
        self,
        config: Optional[Wav2Vec2Config] = None,
        use_audio: bool = True,
        use_visual: bool = True,
        modality_dropout: float = 0.0,
        audio_dropout: float = 0.0,
        fusion_type: str = "concat",  # "concat", "add", "weighted_sum", etc.
        projection_dim: Optional[int] = None, 
        output_dim: Optional[int] = None,
        # Visual encoder specific parameters
        visual_hidden_size: int = 768,
        visual_num_layers: int = 12,
        visual_num_heads: int = 12,
        visual_intermediate_size: int = 3072,
        visual_frontend_channels: int = 64,
        visual_backbone_channels: int = 512,
        use_pretrained_visual_frontend: bool = False,
        visual_frontend_weights_path: Optional[str] = None,
        # Audio encoder specific parameters  
        audio_hidden_size: int = 768,
        audio_num_layers: int = 12,
        audio_num_heads: int = 12,
        audio_intermediate_size: int = 3072,
        audio_conv_dim: List[int] = [512, 512, 512, 512, 512, 512, 512],
        audio_conv_kernel: List[int] = [10, 3, 3, 3, 3, 2, 2],
        audio_conv_stride: List[int] = [5, 2, 2, 2, 2, 2, 2],
    ):
        super().__init__()
        
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.fusion_type = fusion_type
        
        if not (use_audio or use_visual):
            raise ValueError("At least one modality (audio or visual) must be enabled")
        
        # Create config objects if not provided
        if config is None:
            config = Wav2Vec2Config(
                hidden_size=visual_hidden_size if use_visual else audio_hidden_size,
                num_hidden_layers=visual_num_layers if use_visual else audio_num_layers,
                num_attention_heads=visual_num_heads if use_visual else audio_num_heads,
                intermediate_size=visual_intermediate_size if use_visual else audio_intermediate_size,
            )
        
        # Initialize audio encoder if needed
        # --------------------------------------------------------------------------
        # AUDIO ENCODER
        self.audio_encoder = None
        if use_audio:
            self.audio_encoder = AVHuBERTAudioEncoder(
                config=config,
                hidden_size=audio_hidden_size,
                num_hidden_layers=audio_num_layers,
                num_attention_heads=audio_num_heads,
                intermediate_size=audio_intermediate_size,
                conv_dim=audio_conv_dim,
                conv_kernel=audio_conv_kernel,
                conv_stride=audio_conv_stride,
            )
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # VISUAL ENCODER
        # --------------------------------------------------------------------------
        self.visual_encoder = None
        if use_visual:
            self.visual_encoder = AVHuBERTVisualEncoder(
                config=config,
                hidden_size=visual_hidden_size,
                num_hidden_layers=visual_num_layers,
                num_attention_heads=visual_num_heads,
                intermediate_size=visual_intermediate_size,
                frontend_channels=visual_frontend_channels,
                backbone_channels=visual_backbone_channels,
                use_pretrained_frontend=use_pretrained_visual_frontend,
                frontend_weights_path=visual_frontend_weights_path,
            )
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Determine output dimension based on fusion type
        # --------------------------------------------------------------------------
        if output_dim is None:
            if self.fusion_type == "concat" and use_audio and use_visual:
                output_dim = audio_hidden_size + visual_hidden_size
            else:  # For "add", "weighted_sum", or single modality
                output_dim = max(audio_hidden_size if use_audio else 0, 
                                visual_hidden_size if use_visual else 0)
        
        # Optional projection layer after fusion
        self.project_output = None
        if projection_dim is not None and projection_dim > 0:
            self.project_output = nn.Linear(output_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.output_dim = output_dim
        
        # Layer normalization for the output
        self.layer_norm = LayerNorm(self.output_dim)
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,  # (B, T)
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,  # (B, C, T, H, W)
        visual_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        modality_override: Optional[str] = None,  # "audio", "visual", or None for default behavior
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through AV-HuBERT encoder and projection.

        Args:
            audio_input: Audio tensor of shape (B, T)
            audio_attention_mask: Optional mask for audio padding (B, T)
            visual_input: Video frames tensor of shape (B, C, T, H, W)
            visual_attention_mask: Optional mask for video padding (B, T)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            modality_override: Override which modality to use ("audio", "visual", or None)
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state (B, T, output_dim)
                - hidden_states (optional)
                - attentions (optional)
        """
        # Determine which modalities to use
        use_audio = self.use_audio and audio_input is not None
        use_visual = self.use_visual and visual_input is not None
        
        # Apply modality override if provided
        if modality_override == "audio":
            use_audio = self.use_audio and audio_input is not None
            use_visual = False
        elif modality_override == "visual":
            use_audio = False
            use_visual = self.use_visual and visual_input is not None
        
        # Apply modality dropout during training
        if self.training and self.modality_dropout > 0:
            if use_audio and use_visual:  # Only apply dropout if both modalities are available
                dropout_choice = np.random.random()
                if dropout_choice < self.modality_dropout:
                    # Randomly drop one modality
                    if np.random.random() < 0.5:
                        use_audio = False
                    else:
                        use_visual = False
        
        # Apply audio-specific dropout
        if self.training and use_audio and self.audio_dropout > 0:
            if np.random.random() < self.audio_dropout:
                use_audio = False
        
        # Check that at least one modality will be used
        if not (use_audio or use_visual):
            raise ValueError("At least one input modality must be provided and enabled")
        
        # Process audio input if needed
        audio_outputs = None
        if use_audio:
            audio_outputs = self.audio_encoder(
                input_values=audio_input,
                attention_mask=audio_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        # Process visual input if needed
        visual_outputs = None
        if use_visual:
            visual_outputs = self.visual_encoder(
                input_values=visual_input,
                attention_mask=visual_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        # Fuse modalities
        if audio_outputs is not None and visual_outputs is not None:
            # TODO: Handle different sequence lengths by downsampling or upsampling
            # For now, assume the sequence lengths are compatible
            
            if self.fusion_type == "concat":
                # Concatenate along feature dimension
                last_hidden_state = torch.cat([
                    audio_outputs.last_hidden_state, 
                    visual_outputs.last_hidden_state
                ], dim=-1)
            elif self.fusion_type == "add":
                # Simple addition (requires same dimensionality)
                last_hidden_state = audio_outputs.last_hidden_state + visual_outputs.last_hidden_state
            elif self.fusion_type == "weighted_sum":
                # Learnable weighted sum (requires same dimensionality)
                alpha = torch.sigmoid(self.fusion_weight)
                last_hidden_state = alpha * audio_outputs.last_hidden_state + (1 - alpha) * visual_outputs.last_hidden_state
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        elif audio_outputs is not None:
            last_hidden_state = audio_outputs.last_hidden_state
        else:  # visual_outputs is not None
            last_hidden_state = visual_outputs.last_hidden_state
        
        # Apply projection if needed
        if self.project_output is not None:
            last_hidden_state = self.project_output(last_hidden_state)
        
        # Apply layer normalization
        last_hidden_state = self.layer_norm(last_hidden_state)
        
        # Combine other outputs for return
        hidden_states = None
        attentions = None
        
        if output_hidden_states:
            hidden_states = ()
            if audio_outputs is not None and audio_outputs.hidden_states is not None:
                hidden_states += (("audio", audio_outputs.hidden_states),)
            if visual_outputs is not None and visual_outputs.hidden_states is not None:
                hidden_states += (("visual", visual_outputs.hidden_states),)
        
        if output_attentions:
            attentions = ()
            if audio_outputs is not None and audio_outputs.attentions is not None:
                attentions += (("audio", audio_outputs.attentions),)
            if visual_outputs is not None and visual_outputs.attentions is not None:
                attentions += (("visual", visual_outputs.attentions),)
        
        if not return_dict:
            return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        ) 
#==============================================================================



#==============================================================================
#                           AVHuBERTVisualEncoder
# ==============================================================================
class AVHuBERTVisualEncoder(nn.Module):
    """
    Visual Encoder from AV-HuBERT model. This module uses `ResNet` for feature extraction followed by 
    `Transformer Encoder` layers for sequence modeling.

    This is a wrapper for `VisualEncoder` class.
    """
    def __init__(
        self, 
        config: Optional[Wav2Vec2Config] = None,
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
        
        # Use config parameters if provided, otherwise use default arguments
        if config is not None:
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_hidden_layers = getattr(config, "num_hidden_layers", num_hidden_layers)
            num_attention_heads = getattr(config, "num_attention_heads", num_attention_heads)
            intermediate_size = getattr(config, "intermediate_size", intermediate_size)
            hidden_dropout_prob = getattr(config, "hidden_dropout_prob", hidden_dropout_prob) 
            attention_probs_dropout_prob = getattr(config, "attention_probs_dropout_prob", attention_probs_dropout_prob)
            layer_norm_eps = getattr(config, "layer_norm_eps", layer_norm_eps)
            layerdrop = getattr(config, "layerdrop", layerdrop)
        
        # Initialize the VisualEncoder
        self.encoder = VisualEncoderLayer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            frontend_channels=frontend_channels,
            backbone_channels=backbone_channels,
            use_pretrained_frontend=use_pretrained_frontend,
            frontend_weights_path=frontend_weights_path,
            relu_type=relu_type,
            layerdrop=layerdrop,
        )
    
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
            attention_mask: Optional mask for padding (B, T)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state (B, T, hidden_size)
                - hidden_states (optional)
                - attentions (optional)
        """
        return self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
#===============================================================================


#==============================================================================
#                           AVHuBERTAudioEncoder
# ================================================================================
class AVHuBERTAudioEncoder(nn.Module):
    """
    Audio Encoder from AV-HuBERT model. This module processes audio input using 
    convolutional feature extraction followed by transformer encoder layers.

    This is a wrapper for `AudioEncoder` class.
    """
    def __init__(
        self,
        config: Optional[Wav2Vec2Config] = None,
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
        position_embedding_type: str = "conv",
        layerdrop: float = 0.0,
    ):
        super().__init__()
        
        # Use config parameters if provided, otherwise use default arguments
        if config is not None:
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_hidden_layers = getattr(config, "num_hidden_layers", num_hidden_layers)
            num_attention_heads = getattr(config, "num_attention_heads", num_attention_heads)
            intermediate_size = getattr(config, "intermediate_size", intermediate_size)
            hidden_dropout_prob = getattr(config, "hidden_dropout_prob", hidden_dropout_prob)
            attention_probs_dropout_prob = getattr(config, "attention_probs_dropout_prob", attention_probs_dropout_prob)
            layer_norm_eps = getattr(config, "layer_norm_eps", layer_norm_eps)
            layerdrop = getattr(config, "layerdrop", layerdrop)
        
        # Initialize the AudioEncoder
        self.encoder = AudioEncoderLayer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            feat_extract_dropout=feat_extract_dropout,
            feat_extract_activation=feat_extract_activation,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            use_position_embeddings=use_position_embeddings,
            position_embedding_type=position_embedding_type,
            layerdrop=layerdrop,
        )
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through the audio encoder.
        
        Args:
            input_values: Audio tensor of shape (B, T)
            attention_mask: Optional mask for padding (B, T)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            BaseModelOutput with fields:
                - last_hidden_state (B, T, hidden_size)
                - hidden_states (optional)
                - attentions (optional)
        """
        return self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
#==============================================================================