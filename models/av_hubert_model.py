import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass

from transformers import PreTrainedModel, Wav2Vec2Config
from transformers.modeling_outputs import (
    BaseModelOutput,
    # CausalLMOutput, # Use Seq2SeqLMOutput for S2S models
    Seq2SeqLMOutput,
    ModelOutput,
)
from modules.av_hubert_encoder import AVHuBERTEncoderWrapper # THE ENCODER WRAPPER (Wrapping both audio and visual encoders layers)
from modules.av_hubert_decoder import AVHuBERTDecoder # The refactored decoder


# Remove old layer imports if no longer needed
# from modules.av_hubert_layers import LayerNorm, TransformerEncoderLayer # Decoder Layers


@dataclass
class AVHuBERTOutput(ModelOutput):
    """
    Output type of the base AV-HuBERT encoder model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            
    NOTE: The base model is now primarily an encoder, not including loss functions
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AVHuBERTModel(PreTrainedModel):
    """
    The bare AV-HuBERT Model transformer outputting raw hidden-states without any specific head on top.
    This model functions as the encoder in sequence-to-sequence architectures.

    This model inherits from [`PreTrainedModel`] inherited from @huggingface

    Parameters:
        config ([`Wav2Vec2Config`]): Model configuration class defining the model architecture.
            Note that Wav2Vec2Config is used as a base, but should contain AV-HuBERT specific parameters.
        use_audio (`bool`, *optional*, defaults to `True`):
            Whether to use the audio modality encoder.
        use_visual (`bool`, *optional*, defaults to `True`):
            Whether to use the visual modality encoder.
        fusion_type (`str`, *optional*, defaults to `"concat"`):
            How to fuse audio and visual features ("concat", "sum", "film", etc.). Passed to `AVHuBERTEncoderWrapper`.
        encoder_projection_dim (`int`, *optional*):
            Dimension to project fused features to before the main transformer encoder. Passed to `AVHuBERTEncoderWrapper`.
        visual_hidden_size (`int`, *optional*, defaults to 768):
             Dimensionality of the visual encoder layers and the pooler layer. Passed to `AVHuBERTEncoderWrapper`.
        visual_num_layers (`int`, *optional*, defaults to 12):
             Number of hidden layers in the visual Transformer encoder. Passed to `AVHuBERTEncoderWrapper`.
        visual_num_heads (`int`, *optional*, defaults to 12):
             Number of attention heads for each attention layer in the visual Transformer encoder. Passed to `AVHuBERTEncoderWrapper`.
        visual_intermediate_size (`int`, *optional*, defaults to 3072):
             Dimensionality of the "intermediate" (i.e., feed-forward) layer in the visual Transformer encoder. Passed to `AVHuBERTEncoderWrapper`.
        visual_frontend_channels (`int`, *optional*, defaults to 64):
             Output channels of the visual frontend stem. Passed to `AVHuBERTEncoderWrapper`.
        visual_backbone_channels (`int`, *optional*, defaults to 512):
             Output channels of the visual frontend backbone. Passed to `AVHuBERTEncoderWrapper`.
        use_pretrained_visual_frontend (`bool`, *optional*, defaults to `False`):
             Whether to load pretrained weights for the visual frontend ResNet. Passed to `AVHuBERTEncoderWrapper`.
        visual_frontend_weights_path (`str`, *optional*):
             Path to pretrained visual frontend weights if `use_pretrained_visual_frontend` is `True`. Passed to `AVHuBERTEncoderWrapper`.
        mask_time_prob (`float`, *optional*, defaults to 0.065):
            Probability of masking tokens along the time dimension.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of consecutive time steps to mask.
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Probability of masking tokens along the feature dimension.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of consecutive features to mask.
        apply_spec_augment (`bool`, *optional*, defaults to `False`):
            Whether to apply SpecAugment time and feature masking during inference (typically False).
        mask_time_min_masks (`int`, *optional*, defaults to 2):
             Minimum number of time masks applied.
        mask_feature_min_masks (`int`, *optional*, defaults to 0):
             Minimum number of feature masks applied.
        # Removed decoder parameters
        # Removed num_labels (classification head handled by specific `For` models)
    """
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    main_input_name = "audio_input" # Or define based on common use case

    def __init__(
        self,
        config: Optional[Wav2Vec2Config] = None,
        # --------------------------------------------------------------------------
        # Encoder parameters (for AVHuBERTEncoderWrapper)
        use_audio: bool = True,
        use_visual: bool = True,
        fusion_type: str = "concat",
        encoder_projection_dim: Optional[int] = None,
        # additional parameters for visual encoder
        visual_hidden_size: int = 768,
        visual_num_layers: int = 12,
        visual_num_heads: int = 12,
        visual_intermediate_size: int = 3072,
        visual_frontend_channels: int = 64,
        visual_backbone_channels: int = 512,
        use_pretrained_visual_frontend: bool = False,
        visual_frontend_weights_path: Optional[str] = None,
        # --------------------------------------------------------------------------
        # Masking parameters
        mask_time_prob: float = 0.065,
        mask_time_length: int = 10,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        # --------------------------------------------------------------------------
        # SpecAugment parameters for data augmentation
        apply_spec_augment: bool = False,
        mask_time_min_masks: int = 2,
        mask_feature_min_masks: int = 0,
        # Add any other encoder-specific kwargs needed by AVHuBERTEncoderWrapper
        **kwargs # Catch any other relevant config args
    ):
        # --------------------------------------------------------------------------
        # INITIALIZE CONFIG
        if config is None:
            config = Wav2Vec2Config()
            
        config.mask_time_prob = getattr(config, "mask_time_prob", mask_time_prob)
        config.mask_time_length = getattr(config, "mask_time_length", mask_time_length)
        config.mask_feature_prob = getattr(config, "mask_feature_prob", mask_feature_prob)
        config.mask_feature_length = getattr(config, "mask_feature_length", mask_feature_length)
        config.mask_time_min_masks = getattr(config, "mask_time_min_masks", mask_time_min_masks)
        config.mask_feature_min_masks = getattr(config, "mask_feature_min_masks", mask_feature_min_masks)
        config.apply_spec_augment = getattr(config, "apply_spec_augment", apply_spec_augment)

        # Store encoder-specific parameters directly in config if not already present
        # This ensures they are saved and loaded correctly.
        config.use_audio = getattr(config, "use_audio", use_audio)
        config.use_visual = getattr(config, "use_visual", use_visual)
        config.fusion_type = getattr(config, "fusion_type", fusion_type)
        config.encoder_projection_dim = getattr(config, "encoder_projection_dim", encoder_projection_dim)
        config.visual_hidden_size = getattr(config, "visual_hidden_size", visual_hidden_size)
        config.visual_num_layers = getattr(config, "visual_num_layers", visual_num_layers)
        config.visual_num_heads = getattr(config, "visual_num_heads", visual_num_heads)
        config.visual_intermediate_size = getattr(config, "visual_intermediate_size", visual_intermediate_size)
        config.visual_frontend_channels = getattr(config, "visual_frontend_channels", visual_frontend_channels)
        config.visual_backbone_channels = getattr(config, "visual_backbone_channels", visual_backbone_channels)
        config.use_pretrained_visual_frontend = getattr(config, "use_pretrained_visual_frontend", use_pretrained_visual_frontend)
        config.visual_frontend_weights_path = getattr(config, "visual_frontend_weights_path", visual_frontend_weights_path)

        super().__init__(config)

        #------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # INITIALIZE ENCODER
        # Handles either audio-only, visual-only, or audio-visual encoding
        # Pass parameters directly from config
        # --------------------------------------------------------------------------
        self.encoder = AVHuBERTEncoderWrapper(
            config=config,
            use_audio=config.use_audio,
            use_visual=config.use_visual,
            fusion_type=config.fusion_type,
            projection_dim=config.encoder_projection_dim,
            visual_hidden_size=config.visual_hidden_size,
            visual_num_layers=config.visual_num_layers,
            visual_num_heads=config.visual_num_heads,
            visual_intermediate_size=config.visual_intermediate_size,
            visual_frontend_channels=config.visual_frontend_channels,
            visual_backbone_channels=config.visual_backbone_channels,
            use_pretrained_visual_frontend=config.use_pretrained_visual_frontend,
            visual_frontend_weights_path=config.visual_frontend_weights_path,
        )

        # Store encoder output dimension for potential use in heads
        self.hidden_size = self.encoder.output_dim
        # Ensure config.hidden_size matches the actual encoder output dim
        # This is important for downstream models using the config
        self.config.hidden_size = self.hidden_size
        # Also store encoder hidden size for decoder cross-attention
        self.config.encoder_hidden_size = self.hidden_size

        # Initialize weights of the encoder wrapper
        self.init_weights() # Initializes weights of layers defined in this class
    

    #------------------------------------------------------------------------------------------------
    # EXTRACT FEATURES
    # NOTE: This part originally comes from original AVHubert implementation, with some modifications
    #------------------------------------------------------------------------------------------------
    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        modality: str = "audio",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from the model.
        
        Args:
            source: Input source tensor
            padding_mask: Padding mask for the source
            mask: Whether to apply masking
            ret_conv: Whether to return convolutional features
            output_layer: Which transformer layer to output
            modality: Which modality to use ("audio" or "visual")
            
        Returns:
            features: Extracted features
            padding_mask: Updated padding mask
        """
        # This wrapper method adapts to the original interface
        # but delegates to the appropriate encoder method
        
        # Map to Hugging Face interface
        if modality == "audio":
            features = self.encoder.forward(
                audio_input=source,
                audio_attention_mask=padding_mask,
                visual_input=None,
                visual_attention_mask=None,
                mask=mask,
                output_hidden_states=(output_layer is not None),
                modality_override="audio",
                return_dict=True,
            )
        else:  # visual
            features = self.encoder.forward(
                audio_input=None,
                audio_attention_mask=None,
                visual_input=source,
                visual_attention_mask=padding_mask,
                mask=mask,
                output_hidden_states=(output_layer is not None),
                modality_override="visual",
                return_dict=True,
            )
        
        # Get output from specific layer if requested, otherwise use last layer
        if output_layer is not None and features.hidden_states is not None:
            x = features.hidden_states[output_layer]
        else:
            x = features.last_hidden_state
            
        return x, padding_mask
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    # EXTRACT FEATURES FOR FINE-TUNING
    #------------------------------------------------------------------------------------------------
    def extract_finetune(
        self, 
        source, 
        padding_mask=None, 
        mask=False, 
        ret_conv=False, 
        output_layer=None
    ):
        """
        Extract features for fine-tuning. Similar to extract_features but tailored
        for the fine-tuning workflow from the original implementation.
        """
        # This is similar to extract_features but follows the original implementation
        # for fine-tuning

        # Use both modalities if available
        modality_override = None  # Use default modality fusion
        
        features = self.encoder.forward(
            audio_input=source if self.config.use_audio else None,
            audio_attention_mask=padding_mask if self.config.use_audio else None,
            visual_input=source if not self.config.use_audio and self.config.use_visual else None,
            visual_attention_mask=padding_mask if not self.config.use_audio and self.config.use_visual else None,
            mask=mask,
            output_hidden_states=(output_layer is not None),
            modality_override=modality_override,
            return_dict=True,
        )
        
        # Get output from specific layer if requested, otherwise use last layer
        if output_layer is not None and features.hidden_states is not None:
            x = features.hidden_states[output_layer]
        else:
            x = features.last_hidden_state
            
        return x, padding_mask
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    # FORWARD METHOD
    #------------------------------------------------------------------------------------------------
    def forward_gen(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Alternative forward method compatible with the original implementation.
        This provides compatibility with existing training code.
        
        Args:
            source: Audio input tensor
            target_list: List of target tensors (for masked prediction)
            padding_mask: Padding mask for the input
            mask: Whether to apply masking
            features_only: Whether to return only features without loss computation
            output_layer: Specific transformer layer to output
            video: Video input tensor
            
        Returns:
            Dictionary containing model outputs
        """
        # Map parameters to our encoder forward method
        encoder_outputs = self.encoder(
            audio_input=source if self.config.use_audio else None,
            audio_attention_mask=padding_mask,
            visual_input=video if self.config.use_visual else None,
            visual_attention_mask=None,  # No specific mask for video
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            mask=mask,
            modality_override=None,  # Use default fusion behavior
        )
        
        # Get features
        features = encoder_outputs.last_hidden_state
        hidden_states = encoder_outputs.hidden_states
        attentions = encoder_outputs.attentions
        
        # If we only need features, return them directly
        if features_only:
            return {
                "x": features,
                "padding_mask": padding_mask,
                "features": hidden_states,
            }
        
        # For compatibility with original implementation, we should compute loss here
        # But our refactored version handles this in the *ForPreTraining class
        # Return best approximation without computing losses
        return {
            "x": features,
            "padding_mask": padding_mask,
            "features": hidden_states,
            "attn": attentions,
        }
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    #                       ORIGINAL FORWARD METHOD of AVHuBERT
    #------------------------------------------------------------------------------------------------
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # labels: Optional[torch.Tensor] = None, # Labels handled by downstream models
        modality_override: Optional[str] = None,
        apply_masking: bool = True, # Add flag to control masking application
    ) -> Union[Tuple, AVHuBERTOutput]:
        """
        Forward pass for the base AV-HuBERT encoder model.

        Args:
            audio_input (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Float values of input audio waveform.
            audio_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices in audio input.
                Mask values selected in `[0, 1]`: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            visual_input (`torch.Tensor` of shape `(batch_size, channels, time, height, width)`, *optional*):
                Input video frames.
            visual_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices in visual input.
                Corresponds to the time dimension after visual frontend processing.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            modality_override (`str`, *optional*):
                Force the model to use only "audio" or "visual" modality, overriding fusion.
            apply_masking (`bool`, *optional*, defaults to `True`):
                Whether to apply time/feature masking according to config. Should be `True` for pre-training,
                `False` for downstream tasks like ASR/S2S.

        Returns:
            [`AVHuBERTOutput`] or tuple:
            A [`AVHuBERTOutput`] containing the encoder's last hidden state, optional hidden states and attentions,
            or a tuple if `return_dict=False`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # --------------------------------------------------------------------------
        # Pass through encoder wrapper
        # --------------------------------------------------------------------------
        # The encoder wrapper handles modality selection/fusion and returns encoder outputs
        encoder_outputs = self.encoder(
            audio_input=audio_input,
            audio_attention_mask=audio_attention_mask,
            visual_input=visual_input,
            visual_attention_mask=visual_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Ensure the wrapper returns a dict-like object
            modality_override=modality_override,
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state # (B, T_encoded, D)

        # Determine the effective attention mask corresponding to the encoder_hidden_states
        # This mask should reflect padding *after* potential downsampling in the encoder.
        # Need a reliable way to get this mask. Assume the wrapper provides it or we compute it.

        # Option 1: Assume wrapper returns the correct mask `encoder_attention_mask`
        # effective_attention_mask = encoder_outputs.get("attention_mask", None) # Needs wrapper impl.

        # Option 2: Compute it based on input masks and downsampling
        if audio_input is not None and audio_attention_mask is not None:
            # Use audio mask if audio was used
            effective_attention_mask = self._get_feature_vector_attention_mask(
                 encoder_hidden_states.shape[1], audio_attention_mask
            )
        elif visual_input is not None and visual_attention_mask is not None:
             # Use visual mask if visual was used (visual_attention_mask should already be time-aligned)
             # Need to ensure visual_attention_mask corresponds to the output sequence length T_encoded
             # This might require adjustment similar to _get_feature_vector_attention_mask
             # For now, assume visual_attention_mask has the correct length T_encoded
             if visual_attention_mask.shape[1] == encoder_hidden_states.shape[1]:
                  effective_attention_mask = visual_attention_mask
             else:
                  # Recalculate if visual_attention_mask doesn't match output length
                   effective_attention_mask = self._get_feature_vector_attention_mask(
                       encoder_hidden_states.shape[1], visual_attention_mask # Requires original visual time length mask
                   ) # This part is tricky and depends on visual frontend details
                   # Fallback if visual_attention_mask is not directly usable
                   # effective_attention_mask = None # Or generate ones if no mask available
        else:
             # No input mask provided, assume no padding
             effective_attention_mask = torch.ones(encoder_hidden_states.shape[:2],
                                                   dtype=torch.long, device=encoder_hidden_states.device)


        # --------------------------------------------------------------------------
        # Apply masking for pre-training
        # --------------------------------------------------------------------------
        mask_dict = {} # Keep track of applied masks (might be useful for loss calculation)
        hidden_states_to_process = encoder_hidden_states

        # Apply masking only if requested (e.g., during pre-training)
        # Use self.training flag AND the apply_masking argument
        if apply_masking and self.training:
            hidden_states_to_process, mask_dict = self.apply_masking(
                hidden_states=encoder_hidden_states,
                attention_mask=effective_attention_mask, # Use the mask corresponding to hidden states
                training=self.training,
            )

        # The final output of this base model is the processed hidden states from the encoder
        final_hidden_states = hidden_states_to_process

        if not return_dict:
            # Return tuple consistent with BaseModelOutput structure
             outputs = (final_hidden_states,)
             if output_hidden_states:
                 outputs += (encoder_outputs.hidden_states,)
             if output_attentions:
                 outputs += (encoder_outputs.attentions,)
             return tuple(v for v in outputs if v is not None)
        #------------------------------------------------------------------------------------------------

        # Return standard BaseModelOutput or our custom AVHuBERTOutput
        return AVHuBERTOutput(
            last_hidden_state=final_hidden_states,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
            # Include mask dict if needed? Or handle loss in specific ForPreTraining model
        )
    #------------------------------------------------------------------------------------------------
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Calculates the attention mask for the final feature vector length.
        Useful if the encoder downsamples the input. 
        Expand from [batch_size, seq_len] to [batch_size, feature_vector_length]

        Args:
            feature_vector_length: Length of the final feature vector
            attention_mask: Attention mask of the input

        Returns:
            Attention mask of the final feature vector  
        """
        batch_size, seq_len = attention_mask.shape
        # Initialize with ones (default is attend everything)
        output_attention_mask = torch.ones(
            (batch_size, feature_vector_length),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )

        # Compute actual lengths before downsampling
        orig_lengths = attention_mask.sum(dim=-1)

        # Compute expected lengths after potential downsampling by the encoder
        downsampling_ratio = feature_vector_length / seq_len
        if downsampling_ratio.is_integer():
            downsampling_factor = int(downsampling_ratio)
            # Simple division, might need refinement based on actual pooling/conv stride
            new_lengths = (orig_lengths + downsampling_factor - 1) // downsampling_factor
        else:
            # More complex calculation if ratio isn't integer (e.g., from interpolation)
            # Use max length as a safe default if calculation is tricky
            new_lengths = torch.ceil(orig_lengths * (feature_vector_length / seq_len)).long()
            new_lengths = torch.clamp(new_lengths, max=feature_vector_length)


        # Create the mask based on new lengths
        range_tensor = torch.arange(feature_vector_length, device=attention_mask.device)
        # Expand lengths to [batch_size, 1] and compare with range_tensor [feature_vector_length]
        # Resulting mask is True where index < length, False otherwise
        output_attention_mask = range_tensor < new_lengths.unsqueeze(1)
        output_attention_mask = output_attention_mask.long() # Convert boolean to long

        return output_attention_mask

    def _apply_time_masking(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        mask_prob: float = 0.0,
        mask_length: int = 10,
        min_masks: int = 0,
        require_same_masks: bool = True,
        training: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Apply time dimension masking to the input features.

        NOTE: This function is taken from the original AV-HuBERT implementation.

        Args:
            hidden_states: Input features of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask for padding (batch_size, sequence_length)
            mask_prob: Probability of masking a time step
            mask_length: Length of each mask span in time dimension
            min_masks: Minimum number of mask spans
            require_same_masks: Whether all samples in batch should use same masks
            training: Whether in training mode

        Returns:
            Tuple of (masked_features, mask)
        """
        # Return early if no masking needed
        if not training or mask_prob <= 0.0:
            mask = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool, device=hidden_states.device)
            return hidden_states, mask

        batch_size, sequence_length, hidden_size = hidden_states.shape

        # Adjust sequence_length based on attention_mask
        if attention_mask is not None:
            # Compute actual lengths (sum of attention mask values)
            actual_sequence_lengths = attention_mask.sum(-1).to(torch.long)
            # Use the minimum length in batch if masks need to be the same
            valid_length_for_masking = actual_sequence_lengths.min().item() if require_same_masks else sequence_length
        else:
            actual_sequence_lengths = torch.full(
                (batch_size,), sequence_length, device=hidden_states.device, dtype=torch.long
            )
            valid_length_for_masking = sequence_length

        # Ensure mask_length is valid relative to the length used for calculation
        eff_mask_length = min(mask_length, valid_length_for_masking)
        if eff_mask_length < 1:
            return hidden_states, torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool, device=hidden_states.device)

        # Calculate number of masks based on the shortest sequence if require_same_masks=True
        if require_same_masks:
             # Use the shortest sequence length in the batch to determine masking positions
            num_masked_spans = int(mask_prob * valid_length_for_masking / eff_mask_length + np.random.rand())
            num_masked_spans = max(num_masked_spans, min_masks)

            # Ensure we don't mask everything
            if num_masked_spans * eff_mask_length > valid_length_for_masking:
                num_masked_spans = valid_length_for_masking // eff_mask_length
            if num_masked_spans == 0: # Handle cases where valid_length_for_masking is too small
                 return hidden_states, torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool, device=hidden_states.device)


            # Sample mask start indices uniformly from [0, valid_length_for_masking - eff_mask_length]
            mask_idc = np.random.choice(valid_length_for_masking - eff_mask_length + 1, num_masked_spans, replace=False)

            # Create the mask tensor for the whole batch
            mask = torch.zeros((batch_size, sequence_length), device=hidden_states.device, dtype=torch.bool)
            for start_index in mask_idc:
                mask[:, start_index : start_index + eff_mask_length] = True

        else: # Different masks for each sample
            mask = torch.zeros((batch_size, sequence_length), device=hidden_states.device, dtype=torch.bool)
            for i in range(batch_size):
                current_length = actual_sequence_lengths[i].item()
                if current_length == 0: continue # Skip empty sequences

                # Ensure mask length is valid for this specific sequence
                current_eff_mask_length = min(mask_length, current_length)
                if current_eff_mask_length < 1: continue

                num_masked_spans = int(mask_prob * current_length / current_eff_mask_length + np.random.rand())
                num_masked_spans = max(num_masked_spans, min_masks)

                # Ensure we don't mask everything for this sequence
                if num_masked_spans * current_eff_mask_length > current_length:
                     num_masked_spans = current_length // current_eff_mask_length
                if num_masked_spans == 0: continue

                # Sample mask start indices for this sequence
                if current_length - current_eff_mask_length + 1 <= 0: continue # Avoid error if length <= mask_length - 1
                mask_idc = np.random.choice(current_length - current_eff_mask_length + 1, num_masked_spans, replace=False)

                for start_index in mask_idc:
                     mask[i, start_index : start_index + current_eff_mask_length] = True


        # Apply the generated mask, respecting the original padding
        if attention_mask is not None:
            # Ensure mask is only applied where attention_mask is 1
            effective_mask = mask & attention_mask.bool()
        else:
            effective_mask = mask

        # Apply the mask to hidden states (set masked positions to 0, or use a mask embedding if specified in config)
        # Assuming simple zero masking for now
        masked_hidden_states = hidden_states.masked_fill(effective_mask.unsqueeze(-1), 0.0)

        # Return the final mask applied (useful for loss calculation)
        final_mask = effective_mask # Mask indicating True where masking was applied

        return masked_hidden_states, final_mask


    def _apply_feature_masking(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        mask_prob: float = 0.0,
        mask_length: int = 10,
        min_masks: int = 0,
        require_same_masks: bool = True, # Usually True for feature masking
        training: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Apply feature dimension masking to the input features.

        NOTE: This function is taken from the original AV-HuBERT implementation.

        Args:
            hidden_states: Input features of shape (batch_size, sequence_length, hidden_size)
            mask_prob: Probability of masking a feature dimension
            mask_length: Length of each mask span in feature dimension
            min_masks: Minimum number of mask spans
            require_same_masks: Whether all samples in batch should use same masks
            training: Whether in training mode

        Returns:
            Tuple of (masked_features, feature_mask) where feature_mask has shape (batch_size, hidden_size)
        """
         # Return early if no masking needed
        if not training or mask_prob <= 0.0:
            mask = torch.zeros(hidden_states.shape[0], hidden_states.shape[2], dtype=torch.bool, device=hidden_states.device)
            return hidden_states, mask

        batch_size, sequence_length, hidden_size = hidden_states.shape

        # Ensure mask_length is valid
        eff_mask_length = min(mask_length, hidden_size)
        if eff_mask_length < 1:
             return hidden_states, torch.zeros(batch_size, hidden_size, dtype=torch.bool, device=hidden_states.device)

        # Calculate number of masks
        num_masked_spans = int(mask_prob * hidden_size / eff_mask_length + np.random.rand())
        num_masked_spans = max(num_masked_spans, min_masks)

        # Ensure we don't mask everything
        if num_masked_spans * eff_mask_length > hidden_size:
             num_masked_spans = hidden_size // eff_mask_length
        if num_masked_spans == 0:
             return hidden_states, torch.zeros(batch_size, hidden_size, dtype=torch.bool, device=hidden_states.device)


        # Create the feature mask
        if require_same_masks:
            # Sample mask positions - same for all sequences in batch
            if hidden_size - eff_mask_length + 1 <= 0: # Avoid error
                return hidden_states, torch.zeros(batch_size, hidden_size, dtype=torch.bool, device=hidden_states.device)
            mask_idc = np.random.choice(hidden_size - eff_mask_length + 1, num_masked_spans, replace=False)

            # Create mask for one sample and expand
            single_mask = torch.zeros(hidden_size, dtype=torch.bool, device=hidden_states.device)
            for start_index in mask_idc:
                single_mask[start_index : start_index + eff_mask_length] = True
            mask = single_mask.expand(batch_size, -1) # Expand to batch size

        else: # Different masks for each sequence in batch
            mask = torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=torch.bool)
            for i in range(batch_size):
                 # Sample mask positions for this sequence
                if hidden_size - eff_mask_length + 1 <= 0: continue # Avoid error
                mask_idc = np.random.choice(hidden_size - eff_mask_length + 1, num_masked_spans, replace=False)

                 # Apply each mask to this sequence
                for start_index in mask_idc:
                    mask[i, start_index : start_index + eff_mask_length] = True

        # Apply the mask to all time steps
        masked_hidden_states = hidden_states.masked_fill(mask.unsqueeze(1), 0.0)

        return masked_hidden_states, mask # mask is (batch_size, hidden_size)


    def apply_masking(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None, # Mask corresponding to hidden_states dim
        training: bool = True,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.BoolTensor]]:
        """
        Apply time and feature masking based on model config.

        Args:
            hidden_states: Hidden states to mask (B, T, D)
            attention_mask: Attention mask to respect padding for time masking (B, T)
            training: Whether in training mode

        Returns:
            Tuple of (masked_hidden_states, mask_dict)
        """
        mask_dict = {}
        masked_hidden_states = hidden_states

        # Apply time dimension masking if needed
        # Use config attributes directly
        if self.config.mask_time_prob > 0 and (training or self.config.apply_spec_augment):
            masked_hidden_states, time_mask = self._apply_time_masking(
                hidden_states=masked_hidden_states, # Apply sequentially
                attention_mask=attention_mask,
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                min_masks=self.config.mask_time_min_masks,
                # Original code used require_same_masks=False for time masking
                require_same_masks=getattr(self.config, "mask_time_require_same_masks", False),
                training=training,
            )
            mask_dict["time_mask"] = time_mask # Shape (B, T)

        # Apply feature dimension masking if needed
        if self.config.mask_feature_prob > 0 and (training or self.config.apply_spec_augment):
            masked_hidden_states, feature_mask = self._apply_feature_masking(
                hidden_states=masked_hidden_states, # Apply sequentially
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
                 # Original code used require_same_masks=True for feature masking
                require_same_masks=getattr(self.config, "mask_feature_require_same_masks", True),
                training=training,
            )
            mask_dict["feature_mask"] = feature_mask # Shape (B, D)

        return masked_hidden_states, mask_dict


#==============================================================================================
#                           AVHuBERTForCTC
#==============================================================================================
@dataclass
class AVHuBERTForCTCOutput(ModelOutput):
    """
    Output type of [`AVHuBERTForCTC`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Connectionist Temporal Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the vocabulary tokens generated by the TIMIT linear layer.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AVHuBERTForCTC(PreTrainedModel):
    """
    AVHuBERT model with a Connectionist Temporal Classification (CTC) head on top.
    Suitable for tasks like Automatic Speech Recognition (ASR).
    """
    config_class = Wav2Vec2Config # Use base config
    base_model_prefix = "avhubert"

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(config)

        # Initialize base AVHuBERT model (encoder)
        # Pass kwargs which might contain encoder specific params like use_audio etc.
        self.avhubert = AVHuBERTModel(config, **kwargs)

        # CTC head - Ensure vocab_size is present in the config
        if not hasattr(config, "vocab_size"):
             raise ValueError("config must have a vocab_size attribute for AVHuBERTForCTC")
        # The input dimension for the CTC head is the encoder's output dimension
        ctc_input_dim = config.hidden_size

        # Dropout before CTC head
        self.dropout = nn.Dropout(getattr(config, "final_dropout", config.hidden_dropout)) # Use final_dropout or fallback
        # CTC linear layer
        self.lm_head = nn.Linear(ctc_input_dim, config.vocab_size)

        # Initialize weights of the CTC head
        self.init_weights() # Initializes dropout and lm_head

    # Override _init_weights to initialize only the head layers
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # Feature extraction is inside the AVHuBERTEncoderWrapper -> AudioEncoderLayer/VisualEncoderLayer
        # Need to access it through self.avhubert.encoder
        if hasattr(self.avhubert.encoder, 'freeze_feature_extractor'):
             self.avhubert.encoder.freeze_feature_extractor()
        else:
             print("Warning: Encoder wrapper does not have 'freeze_feature_extractor' method.")


    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        modality_override: Optional[str] = None,
    ) -> Union[Tuple, AVHuBERTForCTCOutput]:
        """
        Forward pass for the CTC model.

        Args:
            audio_input (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Float values of input audio waveform.
            audio_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask for audio input.
            visual_input (`torch.Tensor` of shape `(batch_size, channels, time, height, width)`, *optional*):
                Input video frames.
            visual_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask for visual input (time dimension).
            output_attentions (`bool`, *optional*): Whether to return attention weights.
            output_hidden_states (`bool`, *optional*): Whether to return all hidden states.
            return_dict (`bool`, *optional*): Whether to return a dict or tuple.
            labels (`torch.Tensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for computing the CTC loss. Indices are expected in `[0, ..., config.vocab_size - 1]`.
                A value of `-100` corresponds to padding ignored by the loss function.
            modality_override (`str`, *optional*): Force modality usage.

        Returns:
            [`AVHuBERTForCTCOutput`] or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the base AVHuBERT model (encoder)
        # Ensure masking is OFF for CTC inference/fine-tuning
        outputs = self.avhubert(
            audio_input=audio_input,
            audio_attention_mask=audio_attention_mask,
            visual_input=visual_input,
            visual_attention_mask=visual_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modality_override=modality_override,
            apply_masking=False, # <<< Important: No masking for downstream tasks
        )

        # Get the encoder's last hidden state
        hidden_states = outputs.last_hidden_state

        # Apply dropout and CTC head
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Prepare labels: CTC loss expects labels padded with -100
            labels = labels.long() # Ensure labels are long integers
            # Replace padding values if necessary, assuming pad_token_id is used for padding
            # CTC loss expects specific padding (-100) if reduction != 'none'
            # For 'mean' or 'sum' reduction, the loss function handles ignore_index internally.
            # Let's assume labels are already prepared with ignore_index = -100 for padding.

            # CTC loss calculation
            # Input: log_probs (T, B, C), targets (B, S), input_lengths (B,), target_lengths (B,)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1) # (T, B, C)

            # Calculate input lengths from attention mask
            # Need the attention mask corresponding to the logits sequence length
            # Recompute or retrieve the effective_attention_mask from the encoder forward pass
            if audio_input is not None and audio_attention_mask is not None:
                effective_attention_mask = self.avhubert._get_feature_vector_attention_mask(
                    logits.shape[1], audio_attention_mask
                )
            elif visual_input is not None and visual_attention_mask is not None:
                # Assume visual_attention_mask corresponds to logits length or recalculate
                if visual_attention_mask.shape[1] == logits.shape[1]:
                    effective_attention_mask = visual_attention_mask
                else:
                    # Recalculate (this needs careful handling of visual frontend details)
                    effective_attention_mask = self.avhubert._get_feature_vector_attention_mask(
                       logits.shape[1], visual_attention_mask # Needs original visual time length mask
                    )
            else:
                # No mask provided, assume full length
                 effective_attention_mask = torch.ones(logits.shape[:2],
                                                      dtype=torch.long, device=logits.device)

            input_lengths = effective_attention_mask.sum(-1)

            # Calculate target lengths (excluding padding -100)
            # Ensure labels are on the correct device
            labels_mask = labels >= 0 # Mask where labels are not padding
            target_lengths = labels_mask.sum(-1)

            # Flatten labels and remove padding for CTC loss function
            # Note: The built-in CTC loss handles ignore_index, so flattening might not be needed
            # if labels shape is (B, S) and target_lengths is (B,)

            # Set blank token ID - should be defined in config
            blank_id = getattr(self.config, "pad_token_id", 0) # Often PAD token is used as blank

            # Ensure input_lengths and target_lengths are > 0 to avoid CUDA errors
            valid_indices = (input_lengths > 0) & (target_lengths > 0)
            if not valid_indices.all():
                 # Handle cases with zero-length inputs/targets if necessary
                 # Option 1: Skip loss calculation for these samples (might bias results)
                 # Option 2: Return 0 loss for these samples (safer default)
                 # logging.warning("Found samples with zero length input or target for CTC loss.")
                 if valid_indices.any(): # Only compute loss for valid samples
                      loss = F.ctc_loss(
                          log_probs=log_probs[:, valid_indices, :],
                          targets=labels[valid_indices, :],
                          input_lengths=input_lengths[valid_indices],
                          target_lengths=target_lengths[valid_indices],
                          blank=blank_id,
                          reduction="mean", # 'mean' averages over batch and frames
                          zero_infinity=True,
                      )
                 else: # No valid samples
                     loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

            else: # All samples are valid
                 loss = F.ctc_loss(
                     log_probs=log_probs,
                     targets=labels,
                     input_lengths=input_lengths,
                     target_lengths=target_lengths,
                     blank=blank_id,
                     reduction="mean",
                     zero_infinity=True,
                 )


        if not return_dict:
             output = (logits,) + (outputs.hidden_states, outputs.attentions) if return_dict else (outputs[1], outputs[2])
             # Filter None values from tuple
             output = tuple(v for v in output if v is not None)
             return ((loss,) + output) if loss is not None else output

        # Use the dedicated output class
        return AVHuBERTForCTCOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states, # From base model output
            attentions=outputs.attentions,       # From base model output
        )

#============================================================================================



#============================================================================================
#                           AVHuBERTForS2S (Refactored)
#============================================================================================
class AVHuBERTForS2S(PreTrainedModel):
    """
    AVHuBERT model for sequence-to-sequence tasks (e.g., ASR, AST).
    Combines the AVHuBERTModel (encoder) with an AVHuBERTDecoder.
    
    Based on the original AVHubertSeq2Seq implementation from the fairseq repository,
    adapted to work with the Hugging Face Transformers API.
    """
    config_class = Wav2Vec2Config # Use base config, ensure it has decoder params
    base_model_prefix = "avhubert_s2s" # Differentiate prefix
    _tied_weights_keys = ["lm_head.weight", "decoder.embed_tokens.weight"] # Keys for weight tying

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(config)

        # ------------------ Encoder ------------------
        # Instantiate the base AVHuBERTModel as the encoder
        # Pass kwargs which might contain encoder specific params (use_audio etc.)
        self.avhubert = AVHuBERTModel(config, **kwargs)
        
        # ------------------ Decoder ------------------
        # Ensure necessary decoder parameters are in the config
        required_decoder_params = {
            "decoder_embed_dim": getattr(config, "decoder_embed_dim", config.hidden_size),
            "decoder_ffn_embed_dim": getattr(config, "decoder_ffn_dim", 4 * config.hidden_size),
            "decoder_layers": getattr(config, "decoder_layers", 6),
            "decoder_attention_heads": getattr(config, "decoder_attention_heads", config.num_attention_heads),
            "decoder_normalize_before": getattr(config, "decoder_normalize_before", False),
            "decoder_learned_pos": getattr(config, "decoder_learned_pos", False),
            "decoder_layerdrop": getattr(config, "decoder_layerdrop", 0.0),
            "max_target_positions": getattr(config, "max_target_positions", 2048),
            "no_scale_embedding": getattr(config, "no_scale_embedding", True),
            "activation_fn": getattr(config, "activation_function", "gelu"),
            "share_decoder_input_output_embed": getattr(config, "share_decoder_input_output_embed", False),
        }
        
        # Update config with decoder parameters
        for param_name, param_value in required_decoder_params.items():
            setattr(config, param_name, param_value)
            
        # For compatibility with original, set these as class attributes too
        for param_name, param_value in required_decoder_params.items():
            setattr(self, param_name, param_value)
        
        # Make sure d_model is set for decoder
        config.d_model = config.decoder_embed_dim
        # Ensure encoder dim is set for cross-attention
        config.encoder_hidden_size = config.hidden_size
        
        # Setup target dictionary and other parameters needed for decoder
        if not hasattr(config, "vocab_size"):
            raise ValueError("config must have a vocab_size attribute for AVHuBERTForS2S")
        
        # Build embeddings
        self.embed_tokens = self.build_embedding(
            config.vocab_size, config.decoder_embed_dim, config.pad_token_id
        )
            
        # Initialize the decoder with updated config
        decoder_config = config
        self.decoder = AVHuBERTDecoder(decoder_config)
        
        # Set embed_tokens in decoder for proper sharing
        self.decoder.embed_tokens = self.embed_tokens
        
        # ------------------ LM Head ------------------
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        
        # Initialize weights and tie if needed
        self.init_weights()
        
        # Tie embeddings
        if config.share_decoder_input_output_embed:
            self.tie_weights()
    
    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        """
        Build token embeddings similar to original implementation.
        """
        return nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if hasattr(self, "share_decoder_input_output_embed") and self.share_decoder_input_output_embed:
            if getattr(self.config, "torchscript", False):
                # For torchscript, we need to clone weights instead of tying
                self.lm_head.weight = nn.Parameter(self.decoder.embed_tokens.weight.clone())
            else:
                # Standard weight tying
                self.lm_head.weight = self.decoder.embed_tokens.weight
        
    # Accessor methods expected by generate() and other HF utilities
    def get_encoder(self):
        return self.avhubert

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        # Decoder input embeddings
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # Freeze methods (optional but good practice)
    def freeze_encoder(self):
        """Freezes the encoder weights."""
        self.avhubert.eval()
        for param in self.avhubert.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        """Freezes the decoder weights."""
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    # Utility methods for feature extraction
    def extract_features(
        self,
        audio_input=None,
        audio_attention_mask=None,
        visual_input=None,
        visual_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        **kwargs
    ):
        """
        Extract features for encoder-decoder setup.
        Similar to forward but returns decoder outputs directly.
        """
        # Run encoder if encoder_outputs not provided
        if encoder_outputs is None:
            encoder_outputs = self.avhubert(
                audio_input=audio_input,
                audio_attention_mask=audio_attention_mask,
                visual_input=visual_input,
                visual_attention_mask=visual_attention_mask,
                return_dict=True,
                apply_masking=False,  # No masking for inference
            )
        
        # Get encoder attention mask
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = self._prepare_encoder_attention_mask(
            audio_attention_mask if audio_input is not None else visual_attention_mask, 
            encoder_hidden_states
        )
        
        # Run decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=kwargs.get("use_cache", True),
            output_attentions=kwargs.get("output_attentions", False),
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )
        
        return decoder_outputs
    
    def _prepare_encoder_attention_mask(self, input_attention_mask, encoder_hidden_states):
        """
        Prepare the encoder attention mask for use in cross-attention.
        Handles conversion from input mask to encoder output mask.
        """
        if input_attention_mask is None:
            # No mask provided, assume no padding
            return torch.ones(
                encoder_hidden_states.shape[:2], 
                dtype=torch.long, 
                device=encoder_hidden_states.device
            )
            
        # If mask shape matches encoder output shape, use as is
        if input_attention_mask.shape[1] == encoder_hidden_states.shape[1]:
            return input_attention_mask
            
        # Otherwise, need to convert input mask to match encoder output length
        # Use the helper method from AVHuBERTModel
        if hasattr(self.avhubert, "_get_feature_vector_attention_mask"):
            return self.avhubert._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], input_attention_mask
            )
        else:
            # Fallback implementation
            batch_size, seq_len = input_attention_mask.shape
            feature_len = encoder_hidden_states.shape[1]
            
            # Simple ratio-based approach
            ratio = feature_len / seq_len
            if ratio.is_integer():
                # Downsample the mask for integer ratio
                factor = int(ratio)
                new_mask = torch.zeros((batch_size, feature_len), device=input_attention_mask.device)
                for i in range(feature_len):
                    orig_idx = min(i // factor, seq_len - 1)
                    new_mask[:, i] = input_attention_mask[:, orig_idx]
                return new_mask
            else:
                # For non-integer ratio, do simple interpolation
                # This is approximate and may not perfectly match the encoder's downsampling
                orig_lengths = input_attention_mask.sum(-1).float()
                new_lengths = torch.ceil(orig_lengths * ratio).long()
                new_lengths = torch.clamp(new_lengths, max=feature_len)
                
                # Create mask based on calculated lengths
                range_tensor = torch.arange(feature_len, device=input_attention_mask.device)
                # (batch_size, 1) < (1, feature_len) -> (batch_size, feature_len)
                return (range_tensor.unsqueeze(0) < new_lengths.unsqueeze(1)).long()

    def forward(
        self,
        # Encoder inputs
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        # Decoder inputs
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None, 
        # Optional pre-computed encoder outputs
        encoder_outputs: Optional[Union[Tuple, AVHuBERTOutput]] = None,
        # Optional decoder past states
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # Labels for loss calculation
        labels: Optional[torch.LongTensor] = None,
        # Control flags
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_override: Optional[str] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Forward pass for the sequence-to-sequence model.
        
        Args:
            audio_input: Audio waveform input
            audio_attention_mask: Mask for audio input
            visual_input: Video input frames
            visual_attention_mask: Mask for visual input
            decoder_input_ids: Input token IDs for the decoder
            decoder_attention_mask: Attention mask for decoder inputs
            encoder_outputs: Pre-computed encoder outputs
            past_key_values: Past key values for faster decoding
            decoder_inputs_embeds: Pre-computed decoder embeddings
            labels: Target token IDs for computing loss
            use_cache: Whether to use past key values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple.
            modality_override: Force using a specific modality
            
        Returns:
            Seq2SeqLMOutput or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # -------- Handle labels and decoder input IDs --------
        if labels is not None and decoder_input_ids is None:
            # Shift labels to create decoder input IDs
            decoder_input_ids = self._shift_right(labels)
        
        # -------- ENCODER --------
        # Run encoder if outputs not provided
        if encoder_outputs is None:
            encoder_outputs = self.avhubert(
                audio_input=audio_input,
                audio_attention_mask=audio_attention_mask,
                visual_input=visual_input,
                visual_attention_mask=visual_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                modality_override=modality_override,
                apply_masking=False,  # No masking for S2S
            )

        # -------- Prepare encoder outputs for decoder --------
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Prepare encoder attention mask for cross-attention
        encoder_attention_mask = self._prepare_encoder_attention_mask(
            audio_attention_mask if audio_input is not None else visual_attention_mask, 
            encoder_hidden_states
        )
            
        # -------- DECODER --------
        # Prepare decoder inputs
        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # -------- LM Head --------
        decoder_last_hidden_state = decoder_outputs.last_hidden_state
        logits = self.lm_head(decoder_last_hidden_state)

        # -------- Loss Calculation --------
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Move labels to correct device
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # -------- Return --------
        if not return_dict:
            # Return tuple format compatible with Hugging Face
            output = (logits,)
            if use_cache:
                output += (decoder_outputs.past_key_values,)
            if output_hidden_states:
                output += (
                    decoder_outputs.hidden_states,
                    encoder_outputs.hidden_states,
                )
            if output_attentions:
                output += (
                    decoder_outputs.attentions,
                    decoder_outputs.cross_attentions,
                    encoder_outputs.attentions,
                )
            output = (loss,) + output if loss is not None else output
            return output

        # Use standard Seq2SeqLMOutput
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def _shift_right(self, input_ids):
        """
        Shift input_ids to the right for decoder inputs.
        Similar to original implementation in fairseq.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        pad_token_id = self.config.pad_token_id
        
        # Replace beginning of the sequence with decoder start token (usually BOS)
        bos_token_id = getattr(self.config, "bos_token_id", pad_token_id)
        shifted_input_ids[:, 0] = bos_token_id
        
        # Shift input one position to the right
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        
        # Where input_ids was PAD, make the decoder input PAD as well
        shifted_input_ids = torch.where(input_ids == pad_token_id, pad_token_id, shifted_input_ids)
        
        return shifted_input_ids
