# ORIGINAL SOURCE CODE: https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py

import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple, Union, Dict, List, Any

from transformers import Wav2Vec2Config, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

# Local imports
from utils import compute_mask_indices
from modules.av_hubert_layers import (
    AudioEncoderLayer, 
    VisualEncoderLayer,
    LayerNorm,
    GradMultiply
)
# ======================================================================================================
#                           AVHuBERTEncoderWrapper
# ======================================================================================================    
class AVHuBERTEncoderWrapper(PreTrainedModel):
    """
    A wrapper module that manages both audio and visual encoders for AV-HuBERT.
    This module can process audio-only, visual-only, or audio-visual inputs 
    based on the configuration.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    main_input_name = "input_values"
    
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
        # Masking parameters
        mask_prob_audio: float = 0.0,
        mask_prob_image: float = 0.0,
        mask_length_audio: int = 10,
        mask_length_image: int = 10,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        # Feature extraction parameters
        feature_grad_mult: float = 1.0,
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
        super().__init__(config) if config is not None else super().__init__(Wav2Vec2Config())
        
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.fusion_type = fusion_type
        self.feature_grad_mult = feature_grad_mult
        
        # Masking parameters
        self.mask_prob_audio = mask_prob_audio
        self.mask_prob_image = mask_prob_image
        self.mask_length_audio = mask_length_audio
        self.mask_length_image = mask_length_image
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        
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
        #                   USING AUDIO ENCODER
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
        #                   USING VISUAL ENCODER
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
                self.embed = audio_hidden_size + visual_hidden_size
            else:  # For "add", "weighted_sum", or single modality
                self.embed = max(audio_hidden_size if use_audio else 0, 
                               visual_hidden_size if use_visual else 0)
        else:
            self.embed = output_dim
            
        self.encoder_embed_dim = min(audio_hidden_size, visual_hidden_size)
        
        # Optional projection layer after fusion
        self.post_extract_proj = None
        if projection_dim is not None and projection_dim > 0:
            self.post_extract_proj = nn.Linear(self.embed, projection_dim)
            self.output_dim = projection_dim
        elif self.embed != self.encoder_embed_dim:
            self.post_extract_proj = nn.Linear(self.embed, self.encoder_embed_dim)
            self.output_dim = self.encoder_embed_dim
        else:
            self.output_dim = self.embed
        
        # Layer normalization for the output
        self.layer_norm = LayerNorm(self.embed)
        
        # Mask embedding for masking (follow reference implementation)
        self.mask_emb = nn.Parameter(torch.FloatTensor(self.encoder_embed_dim).uniform_())
        
        # Dropout layers
        self.dropout_input = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_features = nn.Dropout(config.hidden_dropout_prob)
    
    # --------------------------------------------------------------------------
    #                   APPLY AUDIO MASK (if AUDIO ENCODER is used)
    def apply_audio_mask(self, x, padding_mask):
        if self.mask_prob_audio > 0:
            B, T = x.shape[:2]
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob_audio,
                self.mask_length_audio,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous()  # [B, C, T]
            x[mask_indices] = self.mask_emb
            x = x.transpose(1, 2).contiguous()  # [B, T, C]
            return x, mask_indices
        return x, None
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    #                   APPLY VISUAL MASK (if VISUAL ENCODER is used)
    def apply_visual_mask(self, x, padding_mask):
        if self.mask_prob_image > 0:
            B, T = x.shape[:2]
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob_image,
                self.mask_length_image,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous()  # [B, C, T]
            x[mask_indices] = self.mask_emb
            x = x.transpose(1, 2).contiguous()  # [B, T, C]
            return x, mask_indices
        return x, None
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        """Extract features with optional gradient multiplication"""
        extractor = self.audio_encoder if modality == "audio" else self.visual_encoder
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,  # (B, T)
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,  # (B, C, T, H, W)
        visual_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        mask: bool = False,  # Whether to apply masking
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
            mask: Whether to apply masking
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
        
        # Apply masking if requested
        if mask:
            if use_audio:
                audio_input, audio_mask_indices = self.apply_audio_mask(audio_input, audio_attention_mask)
            if use_visual:
                visual_input, visual_mask_indices = self.apply_visual_mask(visual_input, visual_attention_mask)
        
        # Apply modality dropout during training
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    use_audio = False
                else:
                    use_visual = False
        
        # Check that at least one modality will be used
        if not (use_audio or use_visual):
            raise ValueError("At least one input modality must be provided and enabled")
        
        # Extract features
        features_audio = None
        features_video = None
        
        if use_audio:
            features_audio = self.forward_features(audio_input, modality="audio")
        
        if use_visual:
            features_video = self.forward_features(visual_input, modality="visual")
        
        # Fuse modalities - following reference implementation
        if features_audio is not None and features_video is not None:
            # Fusion methods from reference
            if self.fusion_type == "concat":
                features = torch.cat([features_audio, features_video], dim=1)
            elif self.fusion_type == "add":
                features = features_audio + features_video
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        elif features_audio is not None:
            features = features_audio
        else:  # features_video is not None
            features = features_video
        
        # Transpose features for layer norm
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        
        # Apply projection if needed
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        # Apply dropout
        features = self.dropout_input(features)
        
        # Prepare final output format
        if not return_dict:
            return (features,)
        
        return BaseModelOutput(
            last_hidden_state=features,
            hidden_states=None,
            attentions=None,
        )



#=======================================================================================================
#                            AVHuBERTAudioEncoder
#=======================================================================================================
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
        
        # Create or update config
        if config is None:
            config = Wav2Vec2Config(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout=hidden_dropout_prob,
                attention_dropout=attention_probs_dropout_prob,
                feat_extract_dropout=feat_extract_dropout,
                feat_extract_activation=feat_extract_activation,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
                conv_stride=conv_stride,
                layerdrop=layerdrop,
                position_embeddings_type=position_embedding_type,
                layer_norm_eps=layer_norm_eps,
            )
        else:
            # Update config with specified parameters
            config.hidden_size = hidden_size
            config.num_hidden_layers = num_hidden_layers
            config.num_attention_heads = num_attention_heads
            config.intermediate_size = intermediate_size
            config.hidden_dropout = hidden_dropout_prob
            config.attention_dropout = attention_probs_dropout_prob
            config.feat_extract_dropout = feat_extract_dropout
            config.feat_extract_activation = feat_extract_activation
            config.layerdrop = layerdrop
            config.layer_norm_eps = layer_norm_eps

        # Create the audio encoder using modified AudioEncoderLayer
        self.encoder = AudioEncoderLayer(config)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass for the audio encoder
        """
        return self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
#-------------------------------------------------------------------------------------------------------------------------------



#======================================================================================================
#                              AVHuBERTVisualEncoder
#======================================================================================================
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
        
        # Create or update config
        if config is None:
            config = Wav2Vec2Config(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout=hidden_dropout_prob,
                attention_dropout=attention_probs_dropout_prob,
                layerdrop=layerdrop,
                layer_norm_eps=layer_norm_eps,
            )
        else:
            # Update config with specified parameters
            config.hidden_size = hidden_size
            config.num_hidden_layers = num_hidden_layers
            config.num_attention_heads = num_attention_heads
            config.intermediate_size = intermediate_size
            config.hidden_dropout = hidden_dropout_prob
            config.attention_dropout = attention_probs_dropout_prob
            config.layerdrop = layerdrop
            config.layer_norm_eps = layer_norm_eps
        
        # Create the visual encoder using VisualEncoderLayer with the modified architecture
        self.encoder = VisualEncoderLayer(
            config=config,
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
        Forward pass for the visual encoder
        """
        return self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
#-------------------------------------------------------------------------------------------------------------------------------