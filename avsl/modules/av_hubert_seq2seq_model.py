"""
AV-HuBERT Sequence-to-Sequence model implementation.
This file implements properly aligned Seq2Seq models for AV-HuBERT for audio-visual speech recognition tasks.
The implementation is based on 2 approaches:
1. Using base `PreTrainedModel` and `ModelOutput` classes: 
    This approach is more flexible and allows for more control over the model architecture. Basically building the model from scratch.

2. Using `Speech2TextModel` and `Speech2TextForConditionalGeneration` classes: 
    These classes are pre-defined by HuggingFace and designed for ASR transcription tasks.

#--------------------------------------------------------------------------------------------------
# NOTE: Using the first approach first, then  fallback to the second approach if needed.
#--------------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from transformers import (
    Speech2TextModel,
    Speech2TextForConditionalGeneration,
    PreTrainedModel, 
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
)
from config.av_hubert_config import AVHuBERTConfig
from modules.av_hubert_model import AVHuBERTModel
from modules.av_hubert_decoder import AVHuBERTDecoder

@dataclass
class AVHuBERTSeq2SeqOutput(ModelOutput):
    """
    Base class for sequence-to-sequence model outputs.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states that can be used to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax.
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class AVHuBERTForSpeech2Text(PreTrainedModel):
    """
    AV-HuBERT model with a sequence-to-sequence architecture for speech-to-text tasks.
    This implementation properly handles both audio and visual modalities.
    
    This model inherits from `PreTrainedModel` and properly initializes both
    encoder and decoder components.
    """
    config_class = AVHuBERTConfig
    base_model_prefix = "av_hubert_seq2seq"
    
    def __init__(self, config: AVHuBERTConfig, **kwargs):
        super().__init__(config)
        
        # Ensure decoder config values are present
        if not hasattr(config, "decoder_layers"):
            config.decoder_layers = getattr(config, "decoder_layers", 6)
        if not hasattr(config, "decoder_attention_heads"):
            config.decoder_attention_heads = getattr(config, "decoder_attention_heads", 8)
        if not hasattr(config, "decoder_ffn_dim"):
            config.decoder_ffn_dim = getattr(config, "decoder_ffn_embed_dim", 2048)
        if not hasattr(config, "activation_function"):
            config.activation_function = getattr(config, "activation_function", "gelu")
        
        # Initialize the encoder
        self.encoder = AVHuBERTModel(config, **kwargs)
        
        # Get encoder output dimension for cross-attention
        encoder_output_dim = self.encoder.hidden_size
        
        # Ensure decoder config knows about encoder dimension
        config.encoder_hidden_size = encoder_output_dim
        
        # Initialize decoder with proper dimensions
        self.decoder = AVHuBERTDecoder(
            config,
            encoder_projection_dim=encoder_output_dim
        )
        
        # Add language model head
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        
        # Weight tying
        if getattr(config, "share_decoder_input_output_embed", False):
            self.lm_head.weight = self.decoder.embed_tokens.weight
        
        # Initialize weights
        self.post_init()
    
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
    
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value
    
    def freeze_encoder(self):
        """Freeze parameters in the encoder."""
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        """Freeze parameters in the decoder."""
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_override: Optional[str] = None,
    ) -> Union[Tuple, AVHuBERTSeq2SeqOutput]:
        """
        Forward pass for the AV-HuBERT sequence-to-sequence model.
        
        Args:
            audio_input: Audio input tensor of shape (batch_size, sequence_length)
            audio_attention_mask: Attention mask for audio input
            visual_input: Visual input tensor of shape (batch_size, channels, frames, height, width)
            visual_attention_mask: Attention mask for visual input
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            encoder_outputs: Pre-computed encoder outputs
            past_key_values: Pre-computed decoder key values for faster inference
            decoder_inputs_embeds: Pre-embedded decoder inputs
            labels: Target labels for computing the loss
            use_cache: Whether to use the past key values cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a ModelOutput object or tuple
            modality_override: Override which modality to use (audio, visual, or None for both)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle both encoder outputs and direct encoding
        if encoder_outputs is None:
            # Run the encoder on audio and/or visual inputs
            encoder_outputs = self.encoder(
                audio_input=audio_input,
                audio_attention_mask=audio_attention_mask,
                visual_input=visual_input,
                visual_attention_mask=visual_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                modality_override=modality_override,
                apply_masking=False,  # No masking for fine-tuning
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Determine effective attention mask for encoder outputs
        if audio_attention_mask is not None:
            encoder_attention_mask = audio_attention_mask
        elif visual_attention_mask is not None:
            encoder_attention_mask = visual_attention_mask
        else:
            # Create attention mask matching encoder hidden states
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], 
                dtype=torch.long, 
                device=encoder_hidden_states.device
            )
        
        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # Shift right for teacher forcing
            decoder_input_ids = self._shift_right(labels)
        
        # Run the decoder
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
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]
        
        # Project to vocabulary
        logits = self.lm_head(sequence_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
        
        if not return_dict:
            outputs = (logits,) + decoder_outputs[1:] + (encoder_outputs,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        
        return AVHuBERTSeq2SeqOutput(
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
        """Shift input tokens right for teacher forcing."""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        # Ensure valid token IDs
        assert decoder_start_token_id is not None, "self.config.decoder_start_token_id must be defined"
        assert pad_token_id is not None, "self.config.pad_token_id must be defined"
        
        # Shift input IDs right by prepending start token and removing last element
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        return shifted_input_ids
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        """
        Prepare inputs for decoder for generation.
        """
        # Cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cached decoder outputs for beam search.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class AVHuBERTForConditionalGeneration(PreTrainedModel):
    """
    AV-HuBERT model with a sequence-to-sequence architecture for conditional text generation.
    This model combines the AVHuBERTModel encoder with the AVHuBERTDecoder and can be used
    for audio-visual speech recognition tasks.
    """
    config_class = AVHuBERTConfig
    base_model_prefix = "av_hubert_seq2seq"
    
    def __init__(self, config: AVHuBERTConfig, **kwargs):
        super().__init__(config)
        
        # Create the Seq2Seq model to use as base
        self.model = AVHuBERTForSpeech2Text(config, **kwargs)
        
        # The LM head is already defined in the model, no need to redefine it
        # Share reference with the already created lm_head
        self.lm_head = self.model.lm_head
        
        # Post initialization
        self.post_init()
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.model.lm_head = new_embeddings
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def freeze_encoder(self):
        self.model.freeze_encoder()
    
    def freeze_decoder(self):
        self.model.freeze_decoder()
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        visual_input: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_override: Optional[str] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Forward pass that delegates to the internal AVHuBERTForSpeech2Text model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through the model
        outputs = self.model(
            audio_input=audio_input,
            audio_attention_mask=audio_attention_mask,
            visual_input=visual_input,
            visual_attention_mask=visual_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modality_override=modality_override,
        )
        
        # Convert to standard HuggingFace Seq2SeqLMOutput if needed
        if return_dict:
            return Seq2SeqLMOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        
        return outputs
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        return self.model.prepare_inputs_for_generation(
            decoder_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            **kwargs
        )
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return AVHuBERTForSpeech2Text._reorder_cache(past_key_values, beam_idx)
#==========================================================================================================================================


    

#==========================================================================================================================================
#                                         AVHuBERT Sequence-to-Sequence Model Implementation 
#                                based on HuggingFace's `Speech2TextModel` and `Speech2TextForConditionalGeneration`
#==========================================================================================================================================

#---------------------------------------------------------------------------------------------  
#                           AVHuBERTSpeech2TextModel
#---------------------------------------------------------------------------------------------
class AVHuBERTSpeech2TextModel(Speech2TextModel):
    config_class = AVHuBERTConfig
    base_model_prefix = "avhubert"

    def __init__(self, config: AVHuBERTConfig, **kwargs):
        super().__init__(config)

        self.avhubert_encoder = AVHuBERTModel(config, **kwargs)
        self.decoder = AVHuBERTDecoder(config)

        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight

    def forward(self, input_features, attention_mask=None, **kwargs):
        return super().forward(input_features, attention_mask, **kwargs)
    
    def get_encoder(self):
        return self.avhubert_encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_output_embeddings(self):
        return self.lm_head
        
    def get_input_embeddings(self):
        return self.decoder.embed_tokens
    
    def freeze_encoder(self):
        self.avhubert_encoder.eval()
        for param in self.avhubert_encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False


#---------------------------------------------------------------------------------------------
#                           AVHuBERTForConditionalGeneration
#---------------------------------------------------------------------------------------------
class AVHuBERTSpeech2TextForConditionalGeneration(Speech2TextForConditionalGeneration):
    config_class = AVHuBERTConfig
    base_model_prefix = "avhubert"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = AVHuBERTSpeech2TextModel(config, **kwargs)
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight
#==========================================================================================================================================
