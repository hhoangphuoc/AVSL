# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#==========================================================================================================================================
#                                         AVHuBERT Configuration Class
#==========================================================================================================================================
from typing import Optional
from transformers import Wav2Vec2Config, PretrainedConfig


class AVHuBERTConfig(PretrainedConfig):
    """
    Configuration class for AVHuBERT model. 
    This configuration inherits from PretrainedConfig and can be used to instantiate a new AVHuBERT model.
    """
    model_type = "av_hubert"
    
    def __init__(
        self,
        # Audio-visual parameters
        use_audio=True,
        use_visual=True,
        fusion_type="concat",  # "concat", "sum", "product"
        encoder_projection_dim=None,  # final projection size after fusion (if None, use default from largest modality)
        
        # Visual encoder params
        visual_hidden_size=768,
        visual_num_hidden_layers=12,
        visual_num_attention_heads=12,
        visual_intermediate_size=3072,
        
        # Audio encoder params - uses Wav2Vec2Config defaults
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        
        # Masking parameters
        mask_time_prob=0.065,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        apply_spec_augment=False,
        
        # Decoder parameters (for S2S)
        decoder_embed_dim=1024,  # Updated from reference model
        decoder_ffn_embed_dim=4096,  # Updated from reference model
        decoder_layers=9,  # Updated from reference model
        decoder_attention_heads=8,  # Updated from reference model
        decoder_layerdrop=0.1,  # Updated from reference model
        decoder_learned_pos=False,
        decoder_normalize_before=True,  # Updated from reference model
        decoder_dropout=0.1,  # New parameter from reference model
        decoder_attention_dropout=0.0,  # New parameter from reference model
        decoder_activation_dropout=0.1,  # New parameter from reference model
        activation_function="gelu",
        
        # Vocabulary & Tokenizer parameters
        vocab_size=10000,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        
        # Embeddings parameters
        no_scale_embedding=True,
        share_decoder_input_output_embed=True,  # Updated from reference model
        max_target_positions=1024,
        
        # Other parameters
        encoder_layerdrop=0.0,
        gradient_checkpointing=False,
        
        # Training parameters
        feature_grad_mult=1.0,
        
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
        
        # Visual encoder parameters
        self.visual_hidden_size = visual_hidden_size
        self.visual_num_hidden_layers = visual_num_hidden_layers
        self.visual_num_attention_heads = visual_num_attention_heads
        self.visual_intermediate_size = visual_intermediate_size
        
        # Audio encoder parameters (from Wav2Vec2)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        
        # Masking parameters
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.apply_spec_augment = apply_spec_augment
        
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
        self.activation_function = activation_function
        
        # Tokenizer parameters
        self.vocab_size = vocab_size
        
        # Embedding parameters
        self.no_scale_embedding = no_scale_embedding
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.max_target_positions = max_target_positions
        
        # Training parameters
        self.feature_grad_mult = feature_grad_mult
        self.encoder_layerdrop = encoder_layerdrop
        self.gradient_checkpointing = gradient_checkpointing
        
        # For compatibility with Wav2Vec2
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
            
        if self.fusion_type == "concat":
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

