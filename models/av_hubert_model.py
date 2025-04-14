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
    CausalLMOutput,
    ModelOutput,
)
from modules.av_hubert_encoder import AVHuBERTEncoderWrapper # THE ENCODER WRAPPER (Wrapping both audio and visual encoders layers)

from modules.av_hubert_layers import LayerNorm, TransformerEncoderLayer # Decoder Layers
from modules.av_hubert_decoder import AVHuBERTDecoder # The decoder

@dataclass
class AVHuBERTOutput(ModelOutput):
    """
    Output type of AV-HuBERT model.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total loss: masked prediction loss + contrastive loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Prediction hidden states output.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
        contrastive_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Contrastive loss.
        masked_lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling loss.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None


class AVHuBERTModel(PreTrainedModel):
    """
    A complete AV-HuBERT model for audio-visual self-supervised learning, pre-training, and fine-tuning.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    
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
        # Decoder parameters (for task-specific layers)
        decoder_hidden_size: Optional[int] = None,
        decoder_num_layers: int = 0,
        decoder_num_attention_heads: int = 0,
        decoder_intermediate_size: int = 3072,  
        # --------------------------------------------------------------------------
        # Final classification parameters
        num_labels: int = 0,  # 0 means no classifier (for pre-training or feature extraction)
        # --------------------------------------------------------------------------
        # SpecAugment parameters for data augmentation
        apply_spec_augment: bool = False,
        mask_time_min_masks: int = 2,
        mask_feature_min_masks: int = 0,
    ):
        if config is None:
            config = Wav2Vec2Config()
            
        # Add relevant parameters to config for later use
        config.mask_time_prob = mask_time_prob
        config.mask_time_length = mask_time_length
        config.mask_feature_prob = mask_feature_prob
        config.mask_feature_length = mask_feature_length
        config.mask_time_min_masks = mask_time_min_masks
        config.mask_feature_min_masks = mask_feature_min_masks
        
        super().__init__(config)
        
        # --------------------------------------------------------------------------
        # Encoder Module
        # Handles either audio-only, visual-only, or audio-visual encoding
        # --------------------------------------------------------------------------
        self.encoder = AVHuBERTEncoderWrapper(
            config=config,
            use_audio=use_audio,
            use_visual=use_visual,
            fusion_type=fusion_type,
            projection_dim=encoder_projection_dim,
            visual_hidden_size=visual_hidden_size,
            visual_num_layers=visual_num_layers,
            visual_num_heads=visual_num_heads,
            visual_intermediate_size=visual_intermediate_size,
            visual_frontend_channels=visual_frontend_channels,
            visual_backbone_channels=visual_backbone_channels,
            use_pretrained_visual_frontend=use_pretrained_visual_frontend,
            visual_frontend_weights_path=visual_frontend_weights_path,
        )
        
        # Store encoder output dimension for later layers
        self.hidden_size = self.encoder.output_dim
        
        # --------------------------------------------------------------------------
        # Masking Settings
        # --------------------------------------------------------------------------
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length  
        self.mask_feature_prob = mask_feature_prob  
        self.mask_feature_length = mask_feature_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_min_masks = mask_feature_min_masks
        self.apply_spec_augment = apply_spec_augment
        
        # --------------------------------------------------------------------------
        # Optional FFN Decoder for task-specific prediction
        # --------------------------------------------------------------------------
        self.decoder_hidden_size = decoder_hidden_size or self.hidden_size
        self.decoder = None
        
        if decoder_num_layers > 0:
            decoder_config = {
                "hidden_size": self.decoder_hidden_size,
                "num_hidden_layers": decoder_num_layers,
                "num_attention_heads": decoder_num_attention_heads,
                "intermediate_size": decoder_intermediate_size,
                "hidden_dropout_prob": config.hidden_dropout_prob,
                "attention_probs_dropout_prob": config.attention_dropout_prob,
                "hidden_act": config.hidden_act,
                "layer_norm_eps": config.layer_norm_eps,
            }
            
            # Create decoder layers
            self.decoder_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    hidden_size=self.decoder_hidden_size,
                    num_attention_heads=decoder_num_attention_heads,
                    intermediate_size=decoder_intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_dropout_prob,
                    hidden_act=config.hidden_act,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(decoder_num_layers)
            ])
            self.decoder_layer_norm = LayerNorm(self.decoder_hidden_size, eps=config.layer_norm_eps)
            
            # Projection from encoder dimension to decoder dimension if they differ
            if self.hidden_size != self.decoder_hidden_size:
                self.encoder_to_decoder_proj = nn.Linear(self.hidden_size, self.decoder_hidden_size)
            else:
                self.encoder_to_decoder_proj = nn.Identity()
        
        # --------------------------------------------------------------------------
        # Optional Classifier Head
        # --------------------------------------------------------------------------
        self.num_labels = num_labels
        if num_labels > 0:
            target_dim = self.decoder_hidden_size if self.decoder is not None else self.hidden_size
            self.classifier = nn.Linear(target_dim, num_labels)
        else:
            self.classifier = None
            
        # Initialize weights
        self.init_weights()
        
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        # Initialize with ones (default is attend everything)
        output_attention_mask = torch.ones(
            (attention_mask.shape[0], feature_vector_length), 
            device=attention_mask.device, 
            dtype=attention_mask.dtype
        )
        
        # Handle zero-padding based on input attention mask
        if attention_mask is not None:
            # Ensure attention_mask is reshaped correctly
            expanded_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, feature_vector_length)
            
            # Adjust dimensions to match feature_vector_length, accounting for possible downsampling
            seq_length = attention_mask.shape[1]
            ratio = feature_vector_length / seq_length
            
            if ratio.is_integer():
                # If exact division (e.g., for pooling operations)
                ratio = int(ratio)
                new_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, ratio)
                new_attention_mask = new_attention_mask.reshape(attention_mask.shape[0], -1)
                output_attention_mask = new_attention_mask
            else:
                # For uneven ratios, use a more complex reshaping approach
                # Example: resize using interpolation
                reshaped_attention_mask = F.interpolate(
                    attention_mask.unsqueeze(1).float(), 
                    size=feature_vector_length, 
                    mode='nearest'
                ).squeeze(1).bool()
                output_attention_mask = reshaped_attention_mask.to(attention_mask.dtype)
        
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
            attention_mask: Optional attention mask for padding
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
            mask = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
            return hidden_states, mask
            
        batch_size, sequence_length, hidden_size = hidden_states.shape
        
        # Adjust sequence_length based on attention_mask
        if attention_mask is not None:
            # Compute actual lengths (sum of attention mask values)
            actual_sequence_lengths = attention_mask.sum(-1).to(torch.long)
            max_length = actual_sequence_lengths.max().item()
        else:
            actual_sequence_lengths = torch.full(
                (batch_size,), sequence_length, device=hidden_states.device, dtype=torch.long
            )
            max_length = sequence_length
        
        # Ensure mask_length is valid
        mask_length = min(mask_length, max_length)
        if mask_length < 1:
            return hidden_states, torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
        
        # Calculate number of masks
        if require_same_masks:
            # Use the shortest sequence for same masks to avoid masking padding
            valid_length = actual_sequence_lengths.min().item()
            num_masked_spans = int(mask_prob * valid_length / mask_length + 0.5)
            num_masked_spans = max(num_masked_spans, min_masks)
            
            # Ensure we don't mask everything
            if num_masked_spans * mask_length > valid_length:
                num_masked_spans = valid_length // mask_length
            
            mask = torch.zeros(
                (batch_size, sequence_length), device=hidden_states.device, dtype=torch.bool
            )
            
            # Determine starting positions for masks
            if num_masked_spans > 0:
                # Sample mask positions
                mask_indices = torch.randperm(
                    valid_length - mask_length + 1, device=hidden_states.device
                )[:num_masked_spans]
                
                # Apply each mask to all batch examples
                for i in range(num_masked_spans):
                    start_index = mask_indices[i]
                    mask[:, start_index:start_index + mask_length] = True
        else:
            # Calculate different masks for each sequence in batch
            mask = torch.zeros(
                (batch_size, sequence_length), device=hidden_states.device, dtype=torch.bool
            )
            
            for i in range(batch_size):
                if actual_sequence_lengths[i] == 0:
                    continue
                    
                valid_length = actual_sequence_lengths[i].item()
                num_masked_spans = int(mask_prob * valid_length / mask_length + 0.5)
                num_masked_spans = max(num_masked_spans, min_masks)
                
                # Ensure we don't mask everything
                if num_masked_spans * mask_length > valid_length:
                    num_masked_spans = valid_length // mask_length
                
                # Sample mask positions for this sequence
                if num_masked_spans > 0:
                    mask_indices = torch.randperm(
                        valid_length - mask_length + 1, device=hidden_states.device
                    )[:num_masked_spans]
                    
                    # Apply each mask to this sequence
                    for j in range(num_masked_spans):
                        start_index = mask_indices[j]
                        mask[i, start_index:start_index + mask_length] = True
        
        # Honor attention mask by not masking padding tokens
        if attention_mask is not None:
            mask = mask * attention_mask.bool()
            
        # Apply the mask to hidden states (set to 0)
        hidden_states = hidden_states.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return hidden_states, mask
    
    def _apply_feature_masking(
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
        Apply feature dimension masking to the input features.

        NOTE: This function is taken from the original AV-HuBERT implementation.
        
        Args:
            hidden_states: Input features of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask for padding
            mask_prob: Probability of masking a feature dimension
            mask_length: Length of each mask span in feature dimension
            min_masks: Minimum number of mask spans
            require_same_masks: Whether all samples in batch should use same masks
            training: Whether in training mode
            
        Returns:
            Tuple of (masked_features, mask)
        """
        # Return early if no masking needed
        if not training or mask_prob <= 0.0:
            mask = torch.zeros_like(hidden_states[:, 0, :], dtype=torch.bool)
            return hidden_states, mask
            
        batch_size, sequence_length, hidden_size = hidden_states.shape
        
        # Ensure mask_length is valid
        mask_length = min(mask_length, hidden_size)
        if mask_length < 1:
            return hidden_states, torch.zeros_like(hidden_states[:, 0, :], dtype=torch.bool)
        
        # Calculate number of masks
        num_masked_spans = int(mask_prob * hidden_size / mask_length + 0.5)
        num_masked_spans = max(num_masked_spans, min_masks)
        
        # Ensure we don't mask everything
        if num_masked_spans * mask_length > hidden_size:
            num_masked_spans = hidden_size // mask_length
            
        # Create the feature mask
        mask = torch.zeros(
            (batch_size, hidden_size), device=hidden_states.device, dtype=torch.bool
        )
        
        if num_masked_spans > 0:
            if require_same_masks:
                # Sample mask positions - same for all sequences in batch
                mask_indices = torch.randperm(
                    hidden_size - mask_length + 1, device=hidden_states.device
                )[:num_masked_spans]
                
                # Apply each mask to all batch examples
                for i in range(num_masked_spans):
                    start_index = mask_indices[i]
                    mask[:, start_index:start_index + mask_length] = True
            else:
                # Different masks for each sequence
                for i in range(batch_size):
                    # Sample mask positions for this sequence
                    mask_indices = torch.randperm(
                        hidden_size - mask_length + 1, device=hidden_states.device
                    )[:num_masked_spans]
                    
                    # Apply each mask to this sequence
                    for j in range(num_masked_spans):
                        start_index = mask_indices[j]
                        mask[i, start_index:start_index + mask_length] = True
        
        # Apply the mask to all time steps
        hidden_states = hidden_states.masked_fill(mask.unsqueeze(1), 0.0)
        
        return hidden_states, mask
    
    def apply_masking(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        training: bool = True,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.BoolTensor]]:
        """
        Apply time and feature masking to hidden states, 
        This function uses `_apply_time_masking` and `_apply_feature_masking` functions for specific masking probabilities.
        
        Args:
            hidden_states: Hidden states to mask (B, T, D)
            attention_mask: Attention mask to respect padding (B, T)
            training: Whether in training mode
            
        Returns:
            Tuple of (masked_hidden_states, mask_dict)
        """
        mask_dict = {}
        
        # Apply time dimension masking if needed
        if self.mask_time_prob > 0 and (training or self.apply_spec_augment):
            hidden_states, time_mask = self._apply_time_masking(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mask_prob=self.mask_time_prob,
                mask_length=self.mask_time_length,
                min_masks=self.mask_time_min_masks,
                require_same_masks=False,  # Use different masks for each sample
                training=training,
            )
            mask_dict["time_mask"] = time_mask
        
        # Apply feature dimension masking if needed
        if self.mask_feature_prob > 0 and (training or self.apply_spec_augment):
            hidden_states, feature_mask = self._apply_feature_masking(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mask_prob=self.mask_feature_prob,
                mask_length=self.mask_feature_length,
                min_masks=self.mask_feature_min_masks,
                require_same_masks=True,  # Use same feature masks for all samples
                training=training,
            )
            mask_dict["feature_mask"] = feature_mask
        
        return hidden_states, mask_dict
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        labels: Optional[torch.Tensor] = None,
        modality_override: Optional[str] = None,
    ) -> Union[Tuple, AVHuBERTOutput]:
        """
        Forward pass for AV-HuBERT model.
        
        Args:
            audio_input: Audio input of shape (batch_size, sequence_length)
            audio_attention_mask: Audio attention mask (batch_size, sequence_length)
            visual_input: Visual input of shape (batch_size, channels, time, height, width)
            visual_attention_mask: Visual attention mask (batch_size, sequence_length)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple
            labels: Optional labels for supervised learning
            modality_override: Override which modality to use ("audio", "visual", or None)
            
        Returns:
            AVHuBERTOutput or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # --------------------------------------------------------------------------
        # Pass through encoder
        # --------------------------------------------------------------------------
        encoder_outputs = self.encoder(
            audio_input=audio_input,
            audio_attention_mask=audio_attention_mask,
            visual_input=visual_input,
            visual_attention_mask=visual_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            modality_override=modality_override,
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Determine the final attention mask (from either audio or visual modality)
        if audio_attention_mask is not None and audio_input is not None:
            attention_mask = audio_attention_mask
        elif visual_attention_mask is not None and visual_input is not None:
            attention_mask = visual_attention_mask
        else:
            attention_mask = None
        
        # --------------------------------------------------------------------------
        # Apply masking for pre-training
        # --------------------------------------------------------------------------
        mask_dict = {}
        
        if self.training:
            encoder_hidden_states, mask_dict = self.apply_masking(
                hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                training=self.training,
            )
        
        # --------------------------------------------------------------------------
        # Apply FFN decoder layers if present
        # --------------------------------------------------------------------------
        hidden_states = encoder_hidden_states
        
        if hasattr(self, "decoder_layers") and len(self.decoder_layers) > 0:
            # Project to decoder dimension if needed
            hidden_states = self.encoder_to_decoder_proj(hidden_states)
            
            # Apply all decoder layers
            for layer in self.decoder_layers:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            
            # Apply layer norm after decoder
            hidden_states = self.decoder_layer_norm(hidden_states)
        
        # --------------------------------------------------------------------------
        # Apply final classifier if needed
        # --------------------------------------------------------------------------
        logits = None
        loss = None
        masked_lm_loss = None
        
        if self.classifier is not None:
            logits = self.classifier(hidden_states)
            
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                
                # Reshape if needed (for sequence classification)
                if logits.shape[0] != labels.shape[0]:
                    raise ValueError(f"Logits and labels batch size mismatch: {logits.shape[0]} vs {labels.shape[0]}")
                
                # For token classification (shape: batch_size, sequence_length, num_classes)
                if len(logits.shape) > 2:
                    # Reshape to (batch_size * sequence_length, num_classes)
                    if attention_mask is not None:
                        # Only compute loss on non-padded tokens
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)[active_loss]
                        active_labels = labels.view(-1)[active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    # For sequence classification (shape: batch_size, num_classes)
                    loss = loss_fct(logits, labels)
                
                masked_lm_loss = loss
        
        if not return_dict:
            return (hidden_states, loss, logits)
        
        return AVHuBERTOutput(
            loss=loss,
            logits=logits if logits is not None else hidden_states,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
            masked_lm_loss=masked_lm_loss,
        )


#==============================================================================================
#                           AVHuBERTForCTC
#==============================================================================================
class AVHuBERTForCTC(PreTrainedModel):
    """
    AVHuBERT model with a CTC head for ASR tasks.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    
    def __init__(
        self,
        config: Wav2Vec2Config,
        **kwargs
    ):
        super().__init__(config)
        
        # Initialize base model
        self.avhubert = AVHuBERTModel(config, **kwargs)
        
        # CTC head
        self.ctc_head = nn.Linear(
            self.avhubert.final_dim, 
            config.vocab_size
        )
        
        # Initialize weights
        self.ctc_head.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass for the CTC model.
        
        Args:
            input_features: Audio input
            attention_mask: Audio mask
            video: Video input
            video_mask: Video mask
            labels: Optional target labels for computing the loss
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            CausalLMOutput with fields:
                - loss (optional, if labels provided)
                - logits
                - hidden_states (optional)
                - attentions (optional)
        """
        outputs = self.avhubert(
            input_features=input_features,
            attention_mask=attention_mask,
            video=video,
            video_mask=video_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask=False,  # No masking for CTC
            return_dict=True,
        )
        
        # Apply CTC head
        logits = self.ctc_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            # Compute CTC loss
            from torch.nn import functional as F
            
            # Ensure padding value for CTC loss is correct
            log_probs = F.log_softmax(logits, dim=-1)
            
            if attention_mask is not None:
                # Sequence length for CTC loss
                input_lengths = attention_mask.sum(-1)
            else:
                input_lengths = torch.ones(logits.shape[0], device=logits.device) * logits.shape[1]
                
            target_lengths = torch.ones(labels.shape[0], device=labels.device) * labels.shape[1]
            
            # CTC loss
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),  # (T, B, C)
                labels,
                input_lengths.int(),
                target_lengths.int(),
                blank=self.config.pad_token_id,
                reduction='mean',
                zero_infinity=True,
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 

#============================================================================================



#============================================================================================
#                           AVHuBERTForS2S
#============================================================================================
    """
    AVHuBERT model for sequence-to-sequence tasks such as speech recognition
    or speech translation. Combines the AVHuBERT encoder with a decoder.
    """
    def __init__(
        self,
        encoder,
        decoder,
        config,
        tie_word_embeddings=True,
    ):
        super().__init__()
        
        self.encoder = encoder #FIXME: This is the encoder wrapper `AVHuBERTEncoderWrapper`
        self.decoder = decoder #FIXME: This is the decoder, refered to `AVHuBERTDecoder`
        self.config = config
        
        # Initialize language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.decoder.embed_tokens.weight
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_input_embeddings(self):
        return self.decoder.embed_tokens
    
    def forward(
        self,
        input_features=None,
        attention_mask=None,
        video=None,
        video_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for the sequence-to-sequence model.
        
        Args:
            input_features: Audio input
            attention_mask: Audio mask
            video: Video input
            video_mask: Video mask
            decoder_input_ids: Input token IDs for the decoder
            decoder_attention_mask: Attention mask for the decoder
            encoder_outputs: Pre-computed encoder outputs
            past_key_values: Cache for faster inference
            decoder_inputs_embeds: Pre-computed decoder embeddings
            labels: Target token IDs
            use_cache: Whether to use the past key values cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Tuple or dict containing:
                - loss (if labels provided)
                - logits
                - encoder_outputs
                - decoder_outputs
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Run encoder if outputs not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                video=video,
                video_mask=video_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Prepare decoder inputs and run decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Project decoder outputs to vocabulary
        logits = self.lm_head(decoder_outputs[0])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            from torch.nn import functional as F
            
            # Move labels to correct device
            labels = labels.to(logits.device)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        # Return appropriate outputs
        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import Seq2SeqLMOutput
        
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
