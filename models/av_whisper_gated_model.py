import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    WhisperConfig, 
    WhisperProcessor, 
    WhisperModel, 
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperEncoder
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from modules.av_hubert_encoder import AVHuBERTEncoderWrapper
from modules.decoders import GatedWhisperDecoder # Assuming GatedWhisperDecoder is in decoder_mods.py

from typing import Optional, Tuple, Union


class AVWhisperGated(nn.Module):
    """
    Main model integrating Whisper Encoder (Audio), AV-HuBERT Encoder (Visual), 
    and a modified Whisper Decoder with Gated Cross Attention.
    """
    def __init__(self, 
                 config: WhisperConfig, 
                 av_hubert_code_path: str, 
                 av_hubert_ckpt_path: str,
                 av_hubert_output_dim: int = 1024, # Default for AV-HuBERT Large
                 freeze_av_hubert: bool = True):
        super().__init__()
        self.config = config

        # --- Instantiate Components --- 

        # 1. Whisper Encoder (Audio)
        # We can load a pretrained Whisper model and extract its encoder,
        # or instantiate directly from config.
        # Using a pretrained model is often easier for initialization.
        # Note: Using WhisperModel encapsulates encoder and gets input_features processing.
        # Alternatively, use WhisperEncoder directly if handling feature extraction separately.
        # Let's use WhisperModel for convenience for now.
        whisper_base = WhisperModel.from_pretrained(config._name_or_path)
        self.audio_encoder = whisper_base.get_encoder()

        # Freeze audio encoder if needed (optional parameter)
        # if freeze_audio_encoder:
        #     for param in self.audio_encoder.parameters():
        #         param.requires_grad = False

        # 2. AV-HuBERT Encoder (Visual)
        self.visual_encoder = AVHuBERTEncoderWrapper(
            av_hubert_code_path=av_hubert_code_path,
            av_hubert_ckpt_path=av_hubert_ckpt_path,
            output_embed_dim=config.hidden_size, # Project to Whisper's hidden dimension
            av_hubert_output_dim=av_hubert_output_dim,
            freeze_encoder=freeze_av_hubert
        )

        # 3. Gated Whisper Decoder
        # Initialize with the same config used for Whisper components
        self.decoder = GatedWhisperDecoder(config)

        # 4. LM Head (Projection to Vocab)
        # Ensure this matches the Whisper standard (usually tied to token embeddings)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size)
        # Consider tying weights: self.proj_out.weight = self.decoder.embed_tokens.weight
        # if config.tie_word_embeddings:
        #     self.proj_out.weight = self.decoder.embed_tokens.weight


    def get_encoder(self):
        # Method needed for compatibility with generate() if inheriting HF PreTrainedModel
        # Here we return the audio encoder, but generation needs careful handling
        # as it now depends on both audio and visual inputs.
        return self.audio_encoder
    
    def get_decoder(self):
        # Method needed for compatibility
        return self.decoder

    def freeze_encoders(self):
        print("Freezing AV-HuBERT encoder.")
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        
        print("Freezing Whisper audio encoder.")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.eval()
        

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None, # Mel Spectrogram (Audio)
        video_features: Optional[torch.FloatTensor] = None, # Video features
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None, # Mask for self-attention
        audio_encoder_attention_mask: Optional[torch.LongTensor] = None, # Padding mask for audio features (if needed by encoder)
        visual_encoder_attention_mask: Optional[torch.LongTensor] = None, # Padding mask for video features
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # Allow passing pre-computed audio features
        visual_encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # Allow passing pre-computed visual features
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if video_features is None and visual_encoder_outputs is None:
             raise ValueError("Video input (`video_features` or `visual_encoder_outputs`) must be provided.")

        # 1. Audio Encoder
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("Audio input (`input_features`) must be provided if `encoder_outputs` is not.")
            # Note: WhisperEncoder expects input_features shape (batch_size, num_mels, seq_len)
            # The WhisperModel forward handles this, but WhisperEncoder needs it directly.
            # If using WhisperEncoder directly, ensure input_features is correctly shaped.
            # Assuming self.audio_encoder is WhisperEncoder instance:
            encoder_outputs = self.audio_encoder(
                 input_features,
                 attention_mask=audio_encoder_attention_mask, # Pass audio padding mask if applicable
                 head_mask=head_mask,
                 output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states,
                 return_dict=return_dict,
             )
        # Ensure encoder_hidden_states is correctly extracted
        audio_encoder_hidden_states = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        # 2. Visual Encoder
        if visual_encoder_outputs is None:
             visual_encoder_hidden_states = self.visual_encoder(
                 video_input=video_features, # Shape assumed (B, T, F) by wrapper
                 video_padding_mask=visual_encoder_attention_mask
             )
             # visual_encoder_outputs structure depends on the wrapper, adjust if needed
             # For now, assume it directly returns the hidden states tensor
        else:
            # If pre-computed, ensure it's the tensor
            if isinstance(visual_encoder_outputs, tuple):
                visual_encoder_hidden_states = visual_encoder_outputs[0]
            else:
                visual_encoder_hidden_states = visual_encoder_outputs 


        # 3. Decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=audio_encoder_attention_mask, # Mask for audio cross-attention
            visual_features=visual_encoder_hidden_states, # Pass visual features
            visual_attention_mask=visual_encoder_attention_mask, # Mask for visual cross-attention
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 4. LM Head
        lm_logits = self.proj_out(decoder_outputs[0]) # Project last hidden state

        # 5. Loss Calculation (Optional)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Shift logits and labels internally if needed (depends on model/task)
            # Typically needed for autoregressive models
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs # Combine outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else None,
            encoder_hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            encoder_attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None,
            # Note: visual encoder outputs are not explicitly part of Seq2SeqLMOutput
        )

# Example Usage (Conceptual)
if __name__ == '__main__':
    # 1. Configuration
    whisper_model_name = "openai/whisper-base" # Or other size
    config = WhisperConfig.from_pretrained(whisper_model_name)
    # Update config if needed (e.g., for dropout in custom layers)
    
    # Paths (replace with your actual paths)
    av_hubert_code_dir = "refs/av_hubert/avhubert" # Path to the dir containing avhubert model defs
    av_hubert_checkpoint = "refs/av_hubert/WEIGHTS/large_noise_pt_noise_ft_433h_only_weights.pt" # Example checkpoint path

    # 2. Instantiate Model
    try:
        model = AVWhisperGated(
            config=config,
            av_hubert_code_path=av_hubert_code_dir, 
            av_hubert_ckpt_path=av_hubert_checkpoint,
            # freeze_av_hubert=True # Optional
        )
        model.eval() # Set to evaluation mode
        print("Model instantiated successfully.")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")

    except FileNotFoundError as e:
         print(f"Error instantiating model: {e}")
         print("Please ensure fairseq is installed and paths to AV-HuBERT code/checkpoint are correct.")
         exit()
    except ImportError as e:
         print(f"Import Error: {e}. Make sure fairseq and transformers are installed.")
         exit()
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         exit()


    # 3. Prepare Dummy Inputs (Replace with actual data loading)
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    batch_size = 2
    seq_len_audio = 3000 # Corresponds to 30s for Whisper features
    seq_len_video = 750  # Example video feature length (e.g., 25fps for 30s)
    seq_len_decoder = 50
    
    # Dummy audio features (Mel Spectrogram)
    # Typically extracted using WhisperFeatureExtractor
    dummy_audio = torch.randn(batch_size, config.num_mel_bins, seq_len_audio)
    
    # Dummy video features (Output from visual frontend/preprocessor)
    # Shape (B, T, FeatureDim) - FeatureDim must match av_hubert_output_dim used in wrapper
    dummy_video = torch.randn(batch_size, seq_len_video, 1024) # Assuming AV-HuBERT Large output dim
    
    # Dummy decoder input IDs (e.g., start token + language token + task token)
    dummy_decoder_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len_decoder))
    # Create dummy labels for loss calculation (shifted version of decoder_ids)
    dummy_labels = dummy_decoder_ids.clone()
    dummy_labels[dummy_labels == processor.tokenizer.pad_token_id] = -100 # Ignore padding in loss

    # Dummy attention masks (1 for real tokens, 0 for padding)
    # Assuming no padding for simplicity here
    dummy_audio_mask = torch.ones(batch_size, seq_len_audio, dtype=torch.long)
    dummy_video_mask = torch.ones(batch_size, seq_len_video, dtype=torch.long)
    # Decoder mask is typically causal + padding mask
    dummy_decoder_mask = torch.ones(batch_size, seq_len_decoder, dtype=torch.long)

    # 4. Forward Pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            input_features=dummy_audio,
            video_features=dummy_video,
            decoder_input_ids=dummy_decoder_ids,
            labels=dummy_labels,
            # visual_encoder_attention_mask=dummy_video_mask, # Pass masks if needed
            return_dict=True
        )

    print("Forward pass successful.")
    print(f"Logits shape: {outputs.logits.shape}") # Should be (batch_size, seq_len_decoder, vocab_size)
    if outputs.loss is not None:
        print(f"Calculated Loss: {outputs.loss.item()}")

    # 5. Generation (Conceptual - requires more setup)
    # Generation with multimodal inputs needs custom handling or 
    # adapting the generate method if inheriting PreTrainedModel.
    # print("\nAttempting generation (conceptual)...")
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    # generated_ids = model.generate(
    #     input_features=dummy_audio[:1], # Generate for first sample
    #     video_features=dummy_video[:1], 
    #     forced_decoder_ids=forced_decoder_ids
    # )
    # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(f"Generated Text: {transcription}") 