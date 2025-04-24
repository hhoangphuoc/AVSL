"""
AVHuBERT configuration module for audio-visual speech recognition model.
"""
from typing import Optional, List
from transformers import Wav2Vec2Config, PretrainedConfig


class AVHuBERTConfig(PretrainedConfig):
    """
    Configuration class for AVHuBERT model. 
    This configuration inherits from PretrainedConfig and can be used to instantiate a new AVHuBERT model.
    
    Configuration values are based on the original AV-HuBERT implementation from Meta Research.
    """
    model_type = "av_hubert"
    
    def __init__(
        self,
        # Audio-visual parameters
        use_audio: bool = True,
        use_visual: bool = True,
        fusion_type: str = "concat",  # "concat", "sum", "weighted_sum"
        encoder_projection_dim: Optional[int] = None,   # Set to None to determine based on fusion_type
        modality_dropout: float = 0.0,                  # Probability of dropping a modality during training
        audio_dropout: float = 0.0,                     # Probability of dropping audio when dropping a modality
        
        # Visual encoder params
        visual_hidden_size: int = 1024,                 # Dimensionality of visual encoder (large model)
        visual_num_hidden_layers: int = 24,             # Number of layers in visual encoder (large model)
        visual_num_attention_heads: int = 16,           # Number of attention heads in visual encoder (large model)
        visual_intermediate_size: int = 4096,           # Size of intermediate layers in visual encoder (large model)
        visual_frontend_channels: int = 64,             # Output channels of visual frontend
        visual_backbone_channels: int = 512,            # Output channels of visual backbone
        use_pretrained_visual_frontend: bool = False,   # Whether to use pretrained visual frontend
        visual_frontend_weights_path: Optional[str] = None,
        
        # Audio encoder params - uses Wav2Vec2Config defaults
        hidden_size: int = 1024,                        # Dimensionality of audio encoder (large model)
        num_hidden_layers: int = 24,                    # Number of layers in audio encoder (large model)
        num_attention_heads: int = 16,                  # Number of attention heads (large model)
        intermediate_size: int = 4096,                  # Size of intermediate layers (large model)
        conv_dim: List[int] = [512, 512, 512, 512, 512, 512, 512],      # Dimensions of CNN layers
        conv_stride: List[int] = [5, 2, 2, 2, 2, 2, 2],                 # Strides of CNN layers
        conv_kernel: List[int] = [10, 3, 3, 3, 3, 2, 2],                # Kernel sizes of CNN layers
        audio_feat_dim: int = 104,                      # Dimension of audio features after preprocessing
        
        # Masking parameters (for fine-tuning)
        mask_time_prob: float = 0.3,                    # Probability for visual time masking
        mask_time_length: int = 5,                      # Length of visual time masking spans
        mask_feature_prob: float = 0.0,                 # Probability for feature masking
        mask_feature_length: int = 10,                  # Length of feature masking spans
        apply_spec_augment: bool = False,               # Whether to apply SpecAugment during inference
        mask_time_min_masks: int = 2,                   # Minimum number of time masks
        mask_feature_min_masks: int = 0,                # Minimum number of feature masks
        
        # Audio and Image masking parameters (for pre-training)
        mask_prob_audio: float = 0.8,                   # Probability for audio time masking
        mask_length_audio: int = 10,                    # Length of audio time masking spans
        mask_prob_image: float = 0.3,                   # Probability for visual time masking
        mask_length_image: int = 5,                     # Length of visual time masking spans

        
        # Dropout parameters
        hidden_dropout: float = 0.1,                   # Dropout in encoder layers
        activation_dropout: float = 0.1,               # Dropout after activation functions
        attention_dropout: float = 0.1,                # Dropout in attention layers
        encoder_layerdrop: float = 0.05,               # Probability of dropping encoder layers
        dropout_input: float = 0.1,                    # Dropout applied to encoder inputs
        dropout_features: float = 0.1,                 # Dropout applied to features
        
        # Seq2Seq decoder parameters
        decoder_embed_dim: int = 1024,                 # Dimensionality of decoder embedding
        decoder_ffn_embed_dim: int = 4096,             # Dimensionality of decoder FFN
        decoder_layers: int = 9,                       # Number of decoder layers
        decoder_attention_heads: int = 8,              # Number of decoder attention heads
        decoder_layerdrop: float = 0.1,                # Probability of dropping decoder layers
        decoder_learned_pos: bool = False,             # Whether to use learned positional embeddings
        decoder_normalize_before: bool = True,         # Whether to use layer norm before attention
        decoder_dropout: float = 0.1,                  # Dropout in decoder
        decoder_attention_dropout: float = 0.0,        # Attention dropout in decoder
        decoder_activation_dropout: float = 0.1,       # Activation dropout in decoder
        
        # Vocabulary & Tokenizer parameters 
        vocab_size: int = 10000,                       # Size of vocabulary
        bos_token_id: int = 0,                         # Beginning of sequence token ID
        pad_token_id: int = 1,                         # Padding token ID
        eos_token_id: int = 2,                         # End of sequence token ID
        
        # Embedding parameters
        no_scale_embedding: bool = True,               # Whether to scale embeddings
        share_decoder_input_output_embed: bool = True, # Share input and output embeddings
        max_target_positions: int = 1024,              # Maximum target sequence length
        
        # Training parameters
        feature_grad_mult: float = 0.1,                # Gradient multiplier for features
        gradient_checkpointing: bool = False,          # Whether to use gradient checkpointing
        label_smoothing: float = 0.1,                  # Label smoothing value
        
        # Other parameters
        layer_norm_first: bool = True,                 # Whether to apply layer norm first in encoder
        final_dim: int = 256,                          # Final projection dimension
        untie_final_proj: bool = True,                 # Whether to untie final projection
        
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Audio-visual parameters
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.fusion_type = fusion_type
        self.encoder_projection_dim = encoder_projection_dim
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        
        # Visual encoder parameters
        self.visual_hidden_size = visual_hidden_size
        self.visual_num_hidden_layers = visual_num_hidden_layers
        self.visual_num_attention_heads = visual_num_attention_heads
        self.visual_intermediate_size = visual_intermediate_size
        self.visual_frontend_channels = visual_frontend_channels
        self.visual_backbone_channels = visual_backbone_channels
        self.use_pretrained_visual_frontend = use_pretrained_visual_frontend
        self.visual_frontend_weights_path = visual_frontend_weights_path
        
        # Audio encoder parameters (from Wav2Vec2)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.audio_feat_dim = audio_feat_dim
        
        # Masking parameters
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_min_masks = mask_feature_min_masks
        
        # Audio-specific masking parameters
        self.mask_time_prob_audio = mask_time_prob_audio
        self.mask_time_length_audio = mask_time_length_audio
        
        # Dropout parameters
        self.hidden_dropout = hidden_dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        
        # Decoder parameters
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_learned_pos = decoder_learned_pos
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.activation_function = kwargs.get("activation_function", "gelu")
        
        # Tokenizer parameters
        self.vocab_size = vocab_size
        
        # Embedding parameters
        self.no_scale_embedding = no_scale_embedding
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.max_target_positions = max_target_positions
        
        # Training parameters
        self.feature_grad_mult = feature_grad_mult
        self.gradient_checkpointing = gradient_checkpointing
        self.label_smoothing = label_smoothing
        
        # Other parameters
        self.layer_norm_first = layer_norm_first
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        
        # For compatibility with HuggingFace models
        self.d_model = decoder_embed_dim
        self.decoder_ffn_dim = decoder_ffn_embed_dim
        
    @property
    def encoder_hidden_size(self):
        """
        Returns the effective encoder hidden size after fusion of modalities.
        """
        if not self.use_audio and not self.use_visual:
            raise ValueError("At least one modality (audio or visual) must be used.")
            
        if self.encoder_projection_dim is not None:
            return self.encoder_projection_dim
            
        if self.fusion_type.lower() == "concat":
            size = 0
            if self.use_audio:
                size += self.hidden_size
            if self.use_visual:
                size += self.visual_hidden_size
            return size
        else:  # "sum", "product" or other fusion types
            # For these fusion types, modalities must have the same dimensions
            # or be projected to the same dimensions
            if self.use_audio and self.use_visual:
                return max(self.hidden_size, self.visual_hidden_size)
            elif self.use_audio:
                return self.hidden_size
            else:  # use_visual
                return self.visual_hidden_size
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Create a configuration object from a YAML file.
        
        Args:
            yaml_path: Path to a YAML configuration file
            
        Returns:
            An instance of AVHuBERTConfig
        """
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Convert nested structure to flat dictionary
            flat_config = {}
            for section_name, section in config_dict.items():
                if isinstance(section, dict):
                    for key, value in section.items():
                        flat_config[f"{section_name}.{key}"] = value
                else:
                    flat_config[section_name] = section
                    
            # Map YAML parameters to config parameters
            mapping = {
                # Audio-visual parameters
                "model.modality_fuse": "fusion_type",
                "model.modality_dropout": "modality_dropout",
                "model.audio_dropout": "audio_dropout",
                
                # Encoder parameters
                "model.encoder_embed_dim": "hidden_size",
                "model.encoder_layers": "num_hidden_layers",
                "model.encoder_attention_heads": "num_attention_heads",
                "model.encoder_ffn_embed_dim": "intermediate_size",
                
                # Use same values for visual encoder by default
                "model.encoder_embed_dim": "visual_hidden_size",
                "model.encoder_layers": "visual_num_hidden_layers",
                "model.encoder_attention_heads": "visual_num_attention_heads",
                "model.encoder_ffn_embed_dim": "visual_intermediate_size",
                
                # Masking parameters
                "model.mask_prob_image": "mask_time_prob",
                "model.mask_length_image": "mask_time_length",
                "model.mask_prob_audio": "mask_time_prob_audio",
                "model.mask_length_audio": "mask_time_length_audio",
                
                # Dropout parameters
                "model.dropout": "hidden_dropout",
                "model.attention_dropout": "attention_dropout",
                "model.activation_dropout": "activation_dropout",
                "model.encoder_layerdrop": "encoder_layerdrop",
                "model.dropout_input": "dropout_input",
                "model.dropout_features": "dropout_features",
                
                # Decoder parameters
                "model.decoder_embed_dim": "decoder_embed_dim",
                "model.decoder_ffn_embed_dim": "decoder_ffn_embed_dim",
                "model.decoder_layers": "decoder_layers",
                "model.decoder_attention_heads": "decoder_attention_heads",
                "model.decoder_layerdrop": "decoder_layerdrop",
                "model.decoder_normalize_before": "decoder_normalize_before",
                "model.decoder_dropout": "decoder_dropout",
                "model.decoder_attention_dropout": "decoder_attention_dropout",
                "model.decoder_activation_dropout": "decoder_activation_dropout",
                
                # Other parameters
                "model.feature_grad_mult": "feature_grad_mult",
                "model.layer_norm_first": "layer_norm_first",
                "model.final_dim": "final_dim",
                "model.untie_final_proj": "untie_final_proj",
                "model.share_decoder_input_output_embed": "share_decoder_input_output_embed",
                
                # Training parameters
                "criterion.label_smoothing": "label_smoothing",
            }
            
            # Create config kwargs from mapping
            config_kwargs = {}
            for yaml_key, config_key in mapping.items():
                if yaml_key in flat_config:
                    config_kwargs[config_key] = flat_config[yaml_key]
            
            # Create config instance
            return cls(**config_kwargs)
        
        except ImportError:
            raise ImportError("To use from_yaml, you need to install the pyyaml package.")
        except Exception as e:
            raise ValueError(f"Error loading YAML configuration: {e}")