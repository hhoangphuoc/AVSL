import os
import sys

# Add project root to path to ensure utils are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)

# Import memory optimization utilities
try:
    from utils.memory_utils import setup_memory_optimizations, clear_gpu_memory, log_memory_stats
    # Apply memory optimizations
    setup_memory_optimizations()
except ImportError:
    # Fallback to direct environment variable setting if utils not available
    print("Warning: Could not import memory_utils. Using fallback memory optimizations.")
    # CRITICAL MEMORY OPTIMIZATIONS - Must be set before importing PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for better performance
    os.environ['TORCH_CUDNN_DISABLE'] = '0'  # Keep cuDNN enabled for performance

#=============================================================================================================================================================
#                                           PATH SETUP AND IMPORTING UTILITIES
#=============================================================================================================================================================  

# Add paths for whisper_flamingo and av_hubert ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_path = os.path.join(project_root, 'utils')  # AVSL/utils
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo')
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert')

sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)  # Add utils path to prioritize AVSL/utils
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)
#----------------------------------------------------------------------------------------------------

# Import fairseq and av_hubert modules ------------------------------------------------------------
# This makes the fairseq modules visible without import conflicts
fairseq_path = os.path.join(av_hubert_path, 'fairseq')
if os.path.exists(fairseq_path) and fairseq_path not in sys.path:
    sys.path.insert(0, fairseq_path)
    print(f"\n✓ Added fairseq path to sys.path: {fairseq_path}\n")

# Set the correct av_hubert path for user module import
# This should point to the avhubert directory within the av_hubert repository
avhubert_user_dir = os.path.join(av_hubert_path, 'avhubert')
print(f"\n✓ AV-HuBERT user dir set to: {avhubert_user_dir}\n")

# Pre-import fairseq to ensure it's loaded correctly
try:
    import fairseq
    print(f"\n✓ Fairseq imported successfully from: {fairseq.__file__}")
    # Verify that the required modules are accessible
    _ = fairseq.checkpoint_utils
    _ = fairseq.utils
    print("✓ Fairseq checkpoint_utils and utils are accessible\n")
except Exception as e:
    print(f"Warning: Fairseq import issue: {e}")
    print("Continuing with the assumption that the original repository fix is in place...")
#----------------------------------------------------------------------------------------------------

# Import video validation utilities -----------------------------------------------------------------
try:
    # Temporarily modify import path to prioritize AVSL/utils
    original_path = sys.path.copy()
    # Remove whisper_flamingo from path temporarily to avoid conflicts
    temp_path = [p for p in sys.path if 'whisper_flamingo' not in p]
    sys.path = temp_path
    
    from utils.hf_video_utils import (
        safe_load_video_feats_from_hf_object,
        debug_hf_video_object,
        create_robust_video_filter,
    )
    
    # Restore original path
    sys.path = original_path
    
    print("\n✅ HuggingFace Video utilities imported successfully\n")
except ImportError as e:
    # Restore original path if error occurred
    if 'original_path' in locals():
        sys.path = original_path
    print(f"\n⚠ Warning: Could not import HF Video utilities: {e}\n")
#----------------------------------------------------------------------------------------------------
# CRITICAL FIX: Add dummy argument to prevent AV-HuBERT duplicate model registration
# This is a known issue with AV-HuBERT: https://github.com/facebookresearch/av_hubert/issues/36
# The workaround is to add a dummy command line argument to prevent the registration conflict
if 'dummy' not in sys.argv and len(sys.argv) == 2:  # Only if we have exactly one argument (the config file)
    print("✓ Adding dummy argument to prevent AV-HuBERT task registration conflicts...")
    sys.argv.append('dummy')
    print(f"✓ sys.argv is now: {sys.argv}")
elif 'dummy' in sys.argv:
    print("✓ Dummy argument already present in sys.argv")
else:
    print(f"✓ sys.argv length is {len(sys.argv)}, no dummy argument modification needed")
#----------------------------------------------------------------------------------------------------



#=============================================================================================================================================================
#                                           IMPORTING UTILITIES
#=============================================================================================================================================================
import yaml
import types
import numpy as np
import torch
from torch import nn
import cv2
import gc  # For garbage collection
# from scipy.io import wavfile # Keep if AmiVideoHFDataset's add_noise needs it via utils
import whisper_flamingo.whisper as whisper
from torchaudio import transforms as T # For potential resampling
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from datasets import load_from_disk # For loading Hugging Face datasets


from whisper_flamingo.spec_augment import spec_augment # Keep if AmiVideoHFDataset uses it directly or via config

from whisper_flamingo.utils import (
    WhisperVideoCollatorWithPadding, # This should still be compatible
    whisper_optimizer,
    whisper_video_projection_optimizer,
    whisper_flamingo_projection_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
    # load_video_feats,
)
from whisper_flamingo.utils_batch_samplers import LengthBatchSampler

# Log memory state before initializing other components
print("🔧 Applying memory optimizations...")
try:
    log_memory_stats("Initial state")
except NameError:
    # If log_memory_stats is not defined, fall back to basic memory reporting
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB allocated")
print("==================================================================")

SAMPLE_RATE = 16000 # Standard Whisper sample rate
AUDIO_HOP_LENGTH = 160 # For calculating number of frames from samples/duration
SEED = 3407
seed_everything(SEED, workers=True)


try:
    from whisper_flamingo.spec_augment import spec_augment
except ImportError:
    print("Warning: Could not import helper functions from utils.py or spec_augment.py. Ensure they are in PYTHONPATH.")


#=============================================================================================================================================================
#                                           DATASET CLASS
#=============================================================================================================================================================
class AmiVideoHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, sample_rate, model_name, 
                 audio_max_length, lang_code="en",
                 spec_augment_config=None, # e.g., "ls-double" or "ls-basic" or None
                 # noise_config=None, # e.g., {\"prob\": 0.5, \"noise_files\": [\"path1\", \"path2\"], \"snr\": 10}
                 train=False):
        super().__init__()

        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.target_sample_rate = sample_rate
        self.model_name = model_name
        self.audio_max_length = audio_max_length # This is whisper.model.N_SAMPLES (e.g., 480000 for 30s)
        self.lang_code = lang_code
        
        self.spec_augment_config = spec_augment_config
        
        self.train = train

        print(f"AmiVideoHFDataset initialized for {'training' if train else 'evaluation'}.")
        print(f"Target sample rate: {self.target_sample_rate}, Audio max length: {self.audio_max_length}")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # 1. Process Audio ----------------------------------------------------------------------------
        audio_data = item['audio']['array'].astype(np.float32)
        source_sample_rate = item['audio']['sampling_rate']

        # Resample if necessary
        if source_sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=source_sample_rate, new_freq=self.target_sample_rate)
            audio_data = resampler(torch.from_numpy(audio_data)).numpy()

        # If it's int16
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0 # Normalize to [-1, 1]
        
        audio = audio_data 
        # --------------------------------------------------------------------------------------------

        audio_frames_before_pad = len(audio.flatten()) // (self.target_sample_rate // 100) # Estimate frames for spec_augment (audio_frames // 160)
        
        # Pad or trim audio
        if self.audio_max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.audio_max_length)
        
        n_mels = 80 if 'large-v3' not in self.model_name else 128 # Compatibility with Whisper models
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels, padding=self.audio_max_length - len(audio) if self.audio_max_length else 0) # freq by time

        # Spec augment ------------------------------------------------------------
        if self.train and self.spec_augment_config and callable(spec_augment) and spec_augment.__name__ != "_dummy_spec_augment":
            if self.spec_augment_config == "ls-double":
                mel_aug_np = spec_augment(mel.T.numpy(), audio_frames=audio_frames_before_pad) # expects time by freq
            elif self.spec_augment_config == "ls-basic":
                mel_aug_np = spec_augment(mel.T.numpy(), audio_frames=audio_frames_before_pad, n_freq_mask=1, n_time_mask=1)
            else:
                raise NotImplementedError
            
            mel = torch.from_numpy(mel_aug_np).T # Transpose back to [freq, time]
        else:
            mel = mel

        # -----------------------------------------------------------------------------------------------


        # 2. Process Text --- WHISPER DECODER ------------------------------------------------------------
        text = item['transcript']
        # For now, we'll keep laugh tokens as text (can be processed later if needed)
        # text = text.replace("[laugh]", " laugh ").replace("(laugh)", " laugh ")

        # Prepare decoder inputs and labels, ensure lang_code is one of tokenizer.all_language_codes
        # Example: self.tokenizer.sot_sequence(language=self.lang_code)
        # Sot sequence with language, task (transcribe), and no_timestamps
        dec_input_ids = [self.tokenizer.sot, 
                         self.tokenizer.special_tokens[f"<|{self.lang_code}|>"], 
                         self.tokenizer.transcribe, 
                         self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text.strip()) # Add leading space as per Whisper common practice
        
        labels = dec_input_ids[1:] + [self.tokenizer.eot] # Shift for teacher forcing
        # --------------------------------------------------------------------------------------------


        # 3. Process Video --- AV HUBERT ------------------------------------------------------------
        # Handle HuggingFace Video object properly
        video_object = item['lip_video']  # This is a HF Video() object
        
        # Debug the video object for first few samples only
        if idx < 2:  # Reduced from 3 to 2 to save memory
            debug_hf_video_object(video_object, idx)
        
        try:
            # Use our improved function to handle HF Video objects
            video_feats = safe_load_video_feats_from_hf_object(
                video_object, 
                train=self.train,
                image_crop_size=88,  # Standard crop size for AV-HuBERT
                image_mean=0.421,
                image_std=0.165
            )
            video_feats = video_feats.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing video for sample {idx}: {e}")
            raise e

        # Trim video to match audio length (approx. 25 fps for video, audio sampling rate for audio)
        # Audio length in seconds = len(audio.flatten()) / self.target_sample_rate
        # Max video frames = (audio length in seconds) * 25 fps
        # MuavicVideoDataset uses: round(len(audio.flatten()) / 16000 * 25)
        # Here, audio is already padded/trimmed to self.audio_max_length
        # So, we should use the original audio length for this calculation if possible, or approximate.
        # Using the actual length of the *processed* audio (which is self.audio_max_length after padding)
        max_video_len_processed_audio = round(len(audio) / self.target_sample_rate * 25)

        if len(video_feats) > max_video_len_processed_audio:
            video_feats = video_feats[:max_video_len_processed_audio]
        
        # Pad video features if they are shorter than a max length, or ensure a consistent size if required by model
        # This part depends on model architecture requirements, MuavicVideoDataset doesn't explicitly show padding here
        # but it might be handled in the collator. For now, just trimming.

        return {
            "input_ids": mel,  # Mel spectrogram (audio features)
            "labels": torch.tensor(labels, dtype=torch.long),
            "dec_input_ids": torch.tensor(dec_input_ids, dtype=torch.long),
            "video": torch.from_numpy(video_feats) # Video features (T, H, W, C)
        }


#=============================================================================================================================================================
#                                           MODEL CLASS
#=============================================================================================================================================================
class WhisperFlamingoModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_hf_dataset, val_hf_dataset, test_hf_dataset) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        #===============================================================================================================
        print("Loading Whisper model =================================================")
        try:
            # Clear memory before loading large model
            try:
                clear_gpu_memory()
                log_memory_stats("Before model load")
            except NameError:
                # Fallback if memory utils aren't available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            self.model = whisper.load_model(model_name,
                                            device='cpu',
                                            download_root=getattr(cfg, 'download_root', None),
                                            dropout_rate=getattr(cfg, 'dropout_rate', 0.0),
                                            video=True,
                                            video_model_path=getattr(cfg, 'video_model_ckpt', None), 
                                            av_hubert_path=avhubert_user_dir,  # Use the correctly set path
                                            prob_av=getattr(cfg, 'prob_use_av', 0.5), 
                                            prob_a=getattr(cfg, 'prob_use_a', 0.5),
                                            av_hubert_encoder=getattr(cfg, 'use_av_hubert_encoder', False),
                                            av_fusion=getattr(cfg, 'av_fusion', 'early'),
                                            add_gated_x_attn=getattr(cfg, 'add_gated_x_attn', 0))
            
            # Enable gradient checkpointing if specified in config
            if getattr(cfg, 'enable_gradient_checkpointing', False):
                try:
                    # Try to use the utility function if available
                    from utils.memory_utils import enable_gradient_checkpointing
                    enable_gradient_checkpointing(self.model)
                except (ImportError, NameError):
                    # Fallback to direct method calls
                    print("✓ Enabling gradient checkpointing for memory optimization")
                    if hasattr(self.model.encoder, 'gradient_checkpointing_enable'):
                        self.model.encoder.gradient_checkpointing_enable()
                    if hasattr(self.model.decoder, 'gradient_checkpointing_enable'):
                        self.model.decoder.gradient_checkpointing_enable()
            
            try:
                log_memory_stats("After model load")
            except NameError:
                pass
                
            print("\n✓ Whisper model loaded successfully\n")
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            print("Please download the model manually or check your internet connection.")
            raise e
        #===============================================================================================================
        
        # Load pre-trained checkpoint if specified
        if cfg.pt_ckpt != '':
            print(f"\n✓ Loading pre-trained audio weights from: {cfg.pt_ckpt}")
            try:
                state_dict = torch.load(cfg.pt_ckpt, map_location=torch.device('cpu'), weights_only=False)
                
                # Handle different checkpoint formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Remove 'model.' prefix if present
                state_dict_updated = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_key = k[6:]  # Remove 'model.'
                    else:
                        new_key = k
                    state_dict_updated[new_key] = v
                
                # Count matching and missing keys
                model_keys = set(self.model.state_dict().keys())
                checkpoint_keys = set(state_dict_updated.keys())
                
                matching_keys = model_keys.intersection(checkpoint_keys)
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                print(f"✓ Checkpoint contains {len(checkpoint_keys)} keys")
                print(f"✓ Model expects {len(model_keys)} keys")
                print(f"✓ Matching keys: {len(matching_keys)}")
                print(f"✓ Missing keys: {len(missing_keys)}")
                print(f"✓ Unexpected keys: {len(unexpected_keys)}")
                
                # Show some missing keys for debugging
                if missing_keys:
                    video_related_missing = [k for k in missing_keys if any(term in k for term in ['video', 'gated_x_attn'])]
                    other_missing = [k for k in missing_keys if not any(term in k for term in ['video', 'gated_x_attn'])]
                    
                    if video_related_missing:
                        print(f"✓ Missing video/gated_x_attn keys: {len(video_related_missing)} (expected for new video components)")
                        if len(video_related_missing) <= 10:
                            print(f"   Examples: {video_related_missing}")
                        else:
                            print(f"   Examples: {video_related_missing[:5]} ... (and {len(video_related_missing)-5} more)")
                    
                    if other_missing:
                        print(f"⚠ Missing non-video keys: {len(other_missing)}")
                        if len(other_missing) <= 10:
                            print(f"   Examples: {other_missing}")
                        else:
                            print(f"   Examples: {other_missing[:5]} ... (and {len(other_missing)-5} more)")
                
                # Try strict loading first, then fallback to non-strict
                try:
                    self.model.load_state_dict(state_dict_updated, strict=True)
                    print("✓ Successfully loaded all weights with strict=True")
                except RuntimeError as e:
                    print(f"✓ Strict loading failed (expected): {str(e)[:200]}...")
                    print("✓ Loading weights with strict=False (ignoring missing video weights)")
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_updated, strict=False)
                    print(f"✓ Successfully loaded compatible weights")
                    if missing_keys:
                        print(f"✓ Initialized {len(missing_keys)} new parameters randomly")
                
            except Exception as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("✓ Continuing with randomly initialized weights...")
                import traceback
                traceback.print_exc() 
        
        self.freeze_video_model = cfg.freeze_video_model
        self.freeze_video_batch_norm_stats = cfg.freeze_video_batch_norm_stats
        
        multilingual = True if 'large' in model_name or '.en' not in model_name else False
        print(f"\n✓ Multilingual tokenizer : {multilingual}, Language for dataset: {lang}\n")

        # --- Use standard Whisper tokenizer (skip custom token for now) ------------------------------------------------------------
        # For now, we'll use the standard Whisper tokenizer without custom tokens
        # Custom tokens can be added later if needed for specific use cases
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=multilingual, language=lang, task='transcribe'
        )
        print(f"\n✓ Using standard Whisper tokenizer with vocab size: {self.tokenizer.encoding.n_vocab}\n")
        # --- End of tokenizer setup ------------------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # Whisper's default ignore_index for padding

        self.cfg = cfg
        self.lang = lang # Store lang for dataset
        
        # Store HF datasets
        self.train_hf_dataset = train_hf_dataset
        self.val_hf_dataset = val_hf_dataset
        self.test_hf_dataset = test_hf_dataset
        
        
        # Duration is in seconds in the HF dataset. Convert to number of frames.
        # Number of frames = (duration_in_seconds * SAMPLE_RATE) / AUDIO_HOP_LENGTH
        # Or more simply, duration_in_seconds * (SAMPLE_RATE / AUDIO_HOP_LENGTH) = duration_in_seconds * 100
        self.train_audio_lengths = [int(d * (SAMPLE_RATE / AUDIO_HOP_LENGTH)) for d in self.train_hf_dataset['duration']]
        self.val_audio_lengths = [int(d * (SAMPLE_RATE / AUDIO_HOP_LENGTH)) for d in self.val_hf_dataset['duration']]
        self.test_audio_lengths = [int(d * (SAMPLE_RATE / AUDIO_HOP_LENGTH)) for d in self.test_hf_dataset['duration']]

        self.special_token_set = set(self.tokenizer.special_tokens.values()) # Update this set with the new tokenizer
        
        # Dataset audio max length (for whisper.pad_or_trim) - typically 30s
        self.dataset_audio_max_length = getattr(cfg, 'dataset_audio_max_length', whisper.audio.N_SAMPLES)


    def forward(self, x): # This method seems unused in original script's training loop
        return self.model(x)

    def training_step(self, batch, batch_id):
        video = batch["video"]
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        padding_mask = batch["padding_mask"] # This comes from WhisperVideoCollatorWithPadding

        if self.freeze_video_model:
            for param in self.model.encoder.video_model.parameters():
                param.requires_grad = False
        if self.freeze_video_batch_norm_stats:
            self.model.encoder.video_model.eval()

        if self.cfg.add_gated_x_attn != 0:
            video_projection_layers = ["video_projection"] if self.cfg.freeze_video_model else ["video"]
            for n, p in self.model.encoder.named_parameters():
                if not any(nd in n for nd in video_projection_layers):
                    p.requires_grad = False
        
        # Clear cache before forward pass, using memory_utils if available
        if batch_id % 10 == 0:  # Clear cache every 10 batches
            try:
                clear_gpu_memory()
            except NameError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        features, x_v = self.model.encoder(input_ids, video, training=True, padding_mask=padding_mask)
        out = self.model.decoder(dec_input_ids, features, xv=x_v)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Log memory usage periodically using memory_utils if available
        if batch_id > 0 and batch_id % 50 == 0:
            try:
                # Get memory stats using our utility function
                from utils.memory_utils import get_memory_stats
                stats = get_memory_stats()
                
                # Log each stat to tensorboard
                for key, value in stats.items():
                    if key.startswith('system'):
                        continue  # Skip system memory for now to avoid cluttering logs
                    self.log(f"train/{key}", value, on_step=True, prog_bar=False, logger=True, sync_dist=False)
            except (ImportError, NameError):
                # Fallback to basic PyTorch memory logging
                if torch.cuda.is_available():
                    self.log("train/gpu_mem_allocated_gb", torch.cuda.memory_allocated() / (1024**3), 
                            on_step=True, prog_bar=False, logger=True, sync_dist=False)
                    self.log("train/gpu_max_mem_allocated_gb", torch.cuda.max_memory_allocated() / (1024**3), 
                            on_step=True, prog_bar=False, logger=True, sync_dist=False)
                    self.log("train/gpu_mem_reserved_gb", torch.cuda.memory_reserved() / (1024**3), 
                            on_step=True, prog_bar=False, logger=True, sync_dist=False)
                    self.log("train/gpu_max_mem_reserved_gb", torch.cuda.max_memory_reserved() / (1024**3), 
                            on_step=True, prog_bar=False, logger=True, sync_dist=False)

        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        video = batch["video"]
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        padding_mask = batch["padding_mask"]

        features_av, x_norm, x_v_norm_pre, x_v_norm_post, x_v = self.model.encoder(input_ids, video, track_norm=True,
                                                                              padding_mask=padding_mask)
        out_av = self.model.decoder(dec_input_ids, features_av, xv=x_v)

        # Original script had separate A-only and V-only paths for validation if add_gated_x_attn == 0
        # This might need adjustment based on how AmiVideoHFDataset and the model handle this.
        # For now, assuming we primarily evaluate the AV model.
        mod_list = {"av": out_av}
        
        # If you need to evaluate A-only and V-only from the same batch, 
        # ensure your model's encoder can be called with flags like `test_a=True` or `test_v=True`
        # and that AmiVideoHFDataset provides inputs compatible with these modes if needed.
        if self.cfg.add_gated_x_attn == 0: # Replicating original logic for A/V modalities
            features_a, _ = self.model.encoder(input_ids, video, test_a=True, padding_mask=padding_mask) # video might be ignored by encoder in test_a mode
            out_a = self.model.decoder(dec_input_ids, features_a)
            mod_list["a"] = out_a

            features_v, x_v_only = self.model.encoder(input_ids, video, test_v=True, padding_mask=padding_mask) # input_ids might be ignored by encoder in test_v mode
            out_v = self.model.decoder(dec_input_ids, features_v, xv=x_v_only) # Pass x_v if relevant for V-only
            mod_list["v"] = out_v


        for mod, out in mod_list.items():
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            tokens = torch.argmax(out, dim=2)

            # Masking tokens after EOT
            eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
            if eot_find.shape[1] > 0 : # Ensure there are tokens to process
                first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1, device=tokens.device) * eot_find, dim=1, keepdim=True)
                token_indices = torch.arange(eot_find.shape[1], device=tokens.device)
                tokens[token_indices.unsqueeze(0) > first_eot] = self.tokenizer.eot
            else: # Handle cases with very short sequences or no EOT found within max length
                 pass


            # Calculate accuracy (excluding special tokens in prompt)
            # Prompt is SOT, LANG, TASK, NO_TIMESTAMPS (4 tokens)
            prompt_len = 4 
            mask = ~(tokens[:, prompt_len:] == self.tokenizer.eot)
            relevant_tokens = tokens[:, prompt_len:].masked_select(mask)
            relevant_labels = labels[:, prompt_len:].masked_select(mask)
            
            n_correct = torch.sum(relevant_tokens.eq(relevant_labels))
            total = torch.sum(mask)
            acc = n_correct.item()  / (total.item() + 1e-6)
            acc = acc if acc <= 1 else 0 # Ensure acc is not > 1 due to empty selections

            o_list, l_list = [], []
            for o_tokens, l_tokens in zip(tokens, labels):
                # Decode ================================================================
                # Filter out negative tokens (like -100 used for ignore_index) and special tokens
                valid_o_tokens = [t for t in o_tokens if t.item() >= 0 and t.item() not in self.special_token_set]
                valid_l_tokens = [t for t in l_tokens if t.item() >= 0 and t.item() not in self.special_token_set]
                
                # Decode the tokens to text
                o_list.append(self.tokenizer.decode(valid_o_tokens))
                l_list.append(self.tokenizer.decode(valid_l_tokens))

            # Calculate WER and CER ================================================================
            wer, cer = wer_cer(hypo=o_list, ref=l_list)
            # =================================================================================
        
            if batch_id < 2 : # Print first 2 batches for inspection
                print(f"Mod: {mod}, Dataloader Idx: {dataloader_idx}")
                for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
                    print("-"*10)
                    print(f"PRED ({i}): {hypo}")
                    print(f"REF  ({i}): {ref}")
                    if i == 1: break # Print 2 examples per modality

            # log_prefix_map = {0: 'test_noisy', 1: 'test_clean', 2: 'val_noisy', 3: 'val_clean'}
            # Simplified logging since there's no noisy variant
            log_prefix_map = {0: 'test', 1: 'val'} 
            log_prefix = log_prefix_map.get(dataloader_idx, f'unknown_idx_{dataloader_idx}')
            
            self.log(f"{log_prefix}/loss_{mod}", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log(f"{log_prefix}/cer_{mod}", cer, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log(f"{log_prefix}/wer_{mod}", wer, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            self.log(f"{log_prefix}/acc_{mod}", acc, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        
        # if dataloader_idx == log_prefix_map.inverse.get('val_clean', 3): # Log norms only for clean validation set
        # Simplified: log norms if it's the validation set (index 1 in the new map)
        if dataloader_idx == 1: # Corresponds to 'val' in the new log_prefix_map
            self.log("val/x_norm", x_norm, prog_bar=False, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_pre", x_v_norm_pre, prog_bar=False, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_post", x_v_norm_post, prog_bar=False, sync_dist=True, add_dataloader_idx=False)
        
        return # For validation_step, no loss needs to be returned for optimization

    def configure_optimizers(self):
        model = self.model
        if self.cfg.add_gated_x_attn != 0:
            optimizer, scheduler = whisper_flamingo_projection_optimizer(model, self.cfg, self.t_total)
        elif self.cfg.video_projection_train_only:
            optimizer, scheduler = whisper_video_projection_optimizer(model, self.cfg, self.t_total)
        else:
            optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def _create_dataloader(self, hf_dataset, audio_lengths, train_mode):
        if train_mode:
            spec_aug_conf = self.cfg.spec_augment
        else: 
            spec_aug_conf = False # No spec augment for val/test

        # Clear memory before creating dataset
        try:
            clear_gpu_memory()
            log_memory_stats(f"Before creating {'training' if train_mode else 'evaluation'} dataset")
        except NameError:
            # Fallback if memory utils aren't available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        dataset = AmiVideoHFDataset(hf_dataset, 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    audio_max_length=self.dataset_audio_max_length, # For pad_or_trim
                                    lang_code=self.lang,
                                    spec_augment_config=spec_aug_conf,
                                    train=train_mode)
        
        # Determine batch size, potentially adapting to available memory
        try:
            from utils.memory_utils import limit_batch_size_to_memory
            
            base_batch_size = self.cfg.batch_size if train_mode else (
                self.cfg.eval_batch_size if hasattr(self.cfg, 'eval_batch_size') else self.cfg.batch_size * 4
            )
            
            # Estimate memory per item based on audio length
            # For simplicity, we'll use a rough estimate
            mem_per_item_gb = 0.25 if 'large' in self.model_name else 0.1
            
            # Potentially adjust batch size based on available memory
            adjusted_batch_size = limit_batch_size_to_memory(
                base_batch_size=base_batch_size,
                required_memory_per_item=mem_per_item_gb,
                safety_factor=0.8
            )
            
            # Calculate batch bins with adjusted batch size
            batch_bins = int(self.cfg.audio_max_length * adjusted_batch_size)
            print(f"Using {'training' if train_mode else 'evaluation'} batch size: {adjusted_batch_size} (from base {base_batch_size})")
            
        except (ImportError, NameError):
            # Fallback to original logic if memory utils aren't available
            batch_bins = int(self.cfg.audio_max_length * (
                self.cfg.batch_size if train_mode else (
                    self.cfg.eval_batch_size if hasattr(self.cfg, 'eval_batch_size') else self.cfg.batch_size * 4
                )
            ))
        
        length_sorter = LengthBatchSampler(batch_bins=batch_bins,
                                           shapes=audio_lengths, # Pre-calculated lengths
                                           sort_in_batch='descending',
                                           sort_batch='shuffle' if train_mode else 'descending',
                                           drop_last=train_mode) # Drop last for training
        if self.cfg.num_devices > 1:
            print("Using distributed sampler for DDP.")
            length_sorter = DistributedSamplerWrapper(length_sorter, shuffle=train_mode) # Shuffle for train

        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=length_sorter,
                                           num_workers=self.cfg.num_worker,
                                           collate_fn=WhisperVideoCollatorWithPadding(),
                                           pin_memory=True, # Added pin_memory
                                           persistent_workers=True if self.cfg.num_worker > 0 else False) # Keep workers alive

    def train_dataloader(self):
        return self._create_dataloader(self.train_hf_dataset, self.train_audio_lengths, train_mode=True)

    def val_dataloader(self):
        # Returns a list of dataloaders for multiple validation sets
        # Order: noisy, clean (to match log_prefix_map if used)
        # val_noisy_loader = self._create_dataloader(self.val_hf_dataset, self.val_audio_lengths, train_mode=False, noise_override_prob=1.0)
        val_clean_loader = self._create_dataloader(self.val_hf_dataset, self.val_audio_lengths, train_mode=False)
        return [val_clean_loader] # New order for log_prefix_map consistency, only clean

    def test_dataloader(self):
        # Returns a list of dataloaders for multiple test sets
        # Order: noisy, clean
        # test_noisy_loader = self._create_dataloader(self.test_hf_dataset, self.test_audio_lengths, train_mode=False, noise_override_prob=1.0)
        test_clean_loader = self._create_dataloader(self.test_hf_dataset, self.test_audio_lengths, train_mode=False)
        return [test_clean_loader]


#=============================================================================================================================================================
#                                           MAIN SCRIPT
#=============================================================================================================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_flamingo_ft_ami.py <config_yaml_path>")
        sys.exit(1)
        
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print("\nConfiguration ================================================================")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print("\n============================================================================\n")

    # Early checks for essential dataset paths from config--------------------------------------------------------------------------------------------------------
    essential_paths_from_config = {
        "train_data_path": getattr(cfg, 'train_data_path', None),
        "val_data_path": getattr(cfg, 'val_data_path', None),
        "test_data_path": getattr(cfg, 'test_data_path', None),
        "train_data_path_original": getattr(cfg, 'train_data_path_original', None),
        "val_data_path_original": getattr(cfg, 'val_data_path_original', None),
        "test_data_path_original": getattr(cfg, 'test_data_path_original', None),
    }
    missing_essential_path = False
    for name, path_val in essential_paths_from_config.items():
        if path_val is None: # Check if attribute exists in config
            print(f"FATAL ERROR: Essential configuration '{name}' is missing in the YAML file.")
            missing_essential_path = True
        elif not os.path.exists(path_val): # Check if path itself exists
            # This check is tricky because the primary paths might be created by the fallback.
            # We rely on the try-except block below for actual loading.
            # However, if _original paths are specified and missing, that's an issue.
            if "_original" in name:
                 print(f"WARNING: Essential original dataset path '{name}': {path_val} not found. Fallback might fail if primary paths also don't exist or can't be created.")
                 # Not setting missing_essential_path to True here, as the script has a fallback.
    if missing_essential_path:
        sys.exit(1)
    #--------------------------------------------------------------------------------------------------------

    print(f"✓ Audio max length for batch sampler: {cfg.audio_max_length} samples")
    dataset_audio_max_len_cfg = getattr(cfg, 'dataset_audio_max_length', whisper.audio.N_SAMPLES)
    print(f"✓ Audio max length for dataset (pad_or_trim): {dataset_audio_max_len_cfg} samples")


    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor)
    
    # Load Hugging Face datasets
    print("\nLoading datasets ============================================================")
    
    def verify_path_exists(path, description):
        """Verify a path exists and provide a helpful error if it doesn't."""
        if not os.path.exists(path):
            print(f"❌ ERROR: {description} path does not exist: {path}")
            return False
        return True
    
    def debug_dataset_sample(dataset, name):
        """Print debug information about a dataset's first sample."""
        try:
            if len(dataset) == 0:
                print(f"⚠️ Warning: {name} dataset is empty")
                return
                
            print(f"\n{name} dataset structure inspection ===============================================")
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if key == 'audio':
                    print(f"  {key}: array shape {value['array'].shape}, sr={value['sampling_rate']}")
                elif key == 'lip_video':
                    print(f"  {key}: {type(value)}")
                else:
                    value_str = str(value)[:100] if isinstance(value, str) else str(type(value))
                    print(f"  {key}: {value_str}")
            print("=============================================================================\n")
        except Exception as e:
            print(f"⚠️ Warning: Error inspecting {name} dataset sample: {e}")
    
    # Function to filter dataset videos
    def filter_dataset_videos(dataset, dataset_name):
        """Filter dataset to remove corrupted videos."""
        print(f"\n📹 Processing {dataset_name} dataset ({len(dataset)} samples)...")
        
        # Use robust video filtering
        valid_indices, corrupted_files = create_robust_video_filter(dataset)
        
        if corrupted_files:
            print(f"🚨 Found {len(corrupted_files)} corrupted videos in {dataset_name}:")
            
            # Show corruption reasons summary
            reasons = {}
            for corrupted in corrupted_files:
                reason = corrupted['reason'].split(':')[0]
                reasons[reason] = reasons.get(reason, 0) + 1
            
            for reason, count in reasons.items():
                print(f"   {reason}: {count} files")
            
            # Show some example files
            if len(corrupted_files) <= 5:
                print(f"   Corrupted files:")
                for corrupted in corrupted_files:
                    print(f"     Index {corrupted['index']}: {corrupted['file']}")
            else:
                print(f"   Example corrupted files (first 3):")
                for corrupted in corrupted_files[:3]:
                    print(f"     Index {corrupted['index']}: {corrupted['file']}")
                print(f"     ... and {len(corrupted_files) - 3} more")
        
        if valid_indices:
            clean_dataset = dataset.select(valid_indices)
            print(f"✅ {dataset_name} clean dataset: {len(clean_dataset)} samples ({len(clean_dataset)/len(dataset)*100:.1f}% retention)")
            return clean_dataset
        else:
            print(f"❌ No valid videos found in {dataset_name} dataset!")
            raise ValueError(f"No valid videos in {dataset_name} dataset")
    
    # Define save paths upfront with validation
    save_dir = os.path.join(project_root, 'data', 'ami')
    train_clean_path = os.path.join(save_dir, "train_clean")
    val_clean_path = os.path.join(save_dir, "val_clean")
    test_clean_path = os.path.join(save_dir, "test_clean")
    
    # First try: Load validated datasets
    datasets_loaded = False
    try:
        print("🔍 TRY: Loading validated videos (already processed)...")
        
        # Check if paths exist before attempting to load
        paths_valid = (
            verify_path_exists(cfg.train_data_path, "Train") and
            verify_path_exists(cfg.val_data_path, "Validation") and
            verify_path_exists(cfg.test_data_path, "Test")
        )
        
        if not paths_valid:
            print("❌ One or more validated dataset paths do not exist. Trying fallback approach.")
            raise FileNotFoundError("One or more validated dataset paths do not exist")
            
        # Load datasets
        print(f"Loading validated train dataset from: {cfg.train_data_path}")
        train_hf_ds = load_from_disk(cfg.train_data_path)
        print(f"✅ Validated train dataset loaded: {len(train_hf_ds)} samples")
        
        print(f"Loading validated validation dataset from: {cfg.val_data_path}")
        val_hf_ds = load_from_disk(cfg.val_data_path)
        print(f"✅ Validated validation dataset loaded: {len(val_hf_ds)} samples")
        
        print(f"Loading validated test dataset from: {cfg.test_data_path}")
        test_hf_ds = load_from_disk(cfg.test_data_path)
        print(f"✅ Validated test dataset loaded: {len(test_hf_ds)} samples")
        
        # Debug dataset structure
        debug_dataset_sample(train_hf_ds, "Train")
        
        # All datasets successfully loaded
        datasets_loaded = True
    
    except FileNotFoundError as e:
        print(f"❌ Path error loading validated datasets: {e}")
        # Will continue to fallback approach
    
    except ImportError as e:
        print(f"❌ Import error loading datasets: {e}")
        print("This may be due to a version mismatch in the datasets library.")
        sys.exit(1)  # Critical error, cannot continue
        
    except ValueError as e:
        print(f"❌ Value error loading datasets: {e}")
        print("This may be due to corrupted dataset files.")
        # Will continue to fallback approach
        
    except Exception as e:
        print(f"❌ Unexpected error loading validated datasets: {e}")
        print(f"Error type: {type(e).__name__}")
        # Will continue to fallback approach
    
    # Fallback approach: Load original datasets and filter them
    if not datasets_loaded:
        try:
            print("\n🔍 TRY FALLBACK APPROACH:")
            print("Step 1: Loading original videos (not processed)...")
            
            # Check if original paths exist before attempting to load
            paths_valid = (
                verify_path_exists(cfg.train_data_path_original, "Train original") and
                verify_path_exists(cfg.val_data_path_original, "Validation original") and
                verify_path_exists(cfg.test_data_path_original, "Test original")
            )
            
            if not paths_valid:
                raise FileNotFoundError("One or more original dataset paths do not exist")
            
            # Load original datasets
            train_hf_ds = load_from_disk(cfg.train_data_path_original)
            val_hf_ds = load_from_disk(cfg.val_data_path_original)
            test_hf_ds = load_from_disk(cfg.test_data_path_original)
            
            print(f"✅ Original train dataset loaded: {len(train_hf_ds)} samples")
            print(f"✅ Original validation dataset loaded: {len(val_hf_ds)} samples")
            print(f"✅ Original test dataset loaded: {len(test_hf_ds)} samples")
    
            print("Step 2: Robust video validation and filtering =================================================")
            print("🔍 Validating videos to identify and remove corrupted files...")
            print("⚠️  This may take a while but ensures stable training...")
        
            # Apply robust video filtering
            try:
                train_hf_ds = filter_dataset_videos(train_hf_ds, "train")
            except Exception as e:
                print(f"❌ Error filtering train dataset: {e}")
                print("Attempting to continue with other datasets...")
            
            try:
                val_hf_ds = filter_dataset_videos(val_hf_ds, "validation")
            except Exception as e:
                print(f"❌ Error filtering validation dataset: {e}")
                print("Attempting to continue with other datasets...")
                
            try:
                test_hf_ds = filter_dataset_videos(test_hf_ds, "test")
            except Exception as e:
                print(f"❌ Error filtering test dataset: {e}")
                print("Attempting to continue with other datasets...")
    
            print("\nDataset sizes after robust filtering ------------------")
            print(f"Train dataset size: {len(train_hf_ds)} samples")
            print(f"Validation dataset size: {len(val_hf_ds)} samples")
            print(f"Test dataset size: {len(test_hf_ds)} samples")
    
            # Save the filtered datasets
            print("\nSaving filtered datasets ------------------------------")
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                train_hf_ds.save_to_disk(train_clean_path)
                print(f"✅ Saved filtered train dataset to {train_clean_path}")
            except Exception as e:
                print(f"❌ Error saving filtered train dataset: {e}")
                
            try:
                val_hf_ds.save_to_disk(val_clean_path)
                print(f"✅ Saved filtered validation dataset to {val_clean_path}")
            except Exception as e:
                print(f"❌ Error saving filtered validation dataset: {e}")
                
            try:
                test_hf_ds.save_to_disk(test_clean_path)
                print(f"✅ Saved filtered test dataset to {test_clean_path}")
            except Exception as e:
                print(f"❌ Error saving filtered test dataset: {e}")
                
            datasets_loaded = True
        
        except FileNotFoundError as e:
            print(f"❌ FATAL ERROR: {e}")
            print("Both primary and fallback dataset paths are missing. Cannot continue.")
            sys.exit(1)
            
        except ImportError as e:
            print(f"❌ FATAL ERROR: Import error loading original datasets: {e}")
            print("This may be due to a version mismatch in the datasets library.")
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ FATAL ERROR: Unexpected error in fallback approach: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Final verification
    if not datasets_loaded or len(train_hf_ds) == 0:
        print("❌ FATAL ERROR: Failed to load datasets or train dataset is empty")
        sys.exit(1)
    
    print("\n===============================================================================================\n")


    # 2. Filter by duration (Second Filtering) ------------------------------------------------------------
    print("\n 2. Filter by duration (Second Filtering) ======================================================")
    max_duration_filter_sec = getattr(cfg, 'max_duration_filter_seconds', 30.0) # Max 30s by default
    
    # Filter by duration first (before robust video validation)
    print(f"\nFiltering by duration (max {max_duration_filter_sec}s)--------------------")
    original_train_size = len(train_hf_ds)
    original_val_size = len(val_hf_ds) 
    original_test_size = len(test_hf_ds)
    
    train_hf_ds = train_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    val_hf_ds = val_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    test_hf_ds = test_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    
    print(f"After duration filtering:")
    print(f"  Train: {original_train_size} → {len(train_hf_ds)} ({len(train_hf_ds)/original_train_size*100:.1f}%)")
    print(f"  Val: {original_val_size} → {len(val_hf_ds)} ({len(val_hf_ds)/original_val_size*100:.1f}%)")
    print(f"  Test: {original_test_size} → {len(test_hf_ds)} ({len(test_hf_ds)/original_test_size*100:.1f}%)")
    print("\n===============================================================================================\n")

    # Check for empty datasets after all filtering
    if len(train_hf_ds) == 0:
        print("FATAL ERROR: Training dataset is empty after filtering. Training cannot proceed.")
        sys.exit(1)
    if len(val_hf_ds) == 0:
        print("WARNING: Validation dataset is empty after filtering.")
    if len(test_hf_ds) == 0:
        print("WARNING: Test dataset is empty after filtering.")


    # 3. Dataset Slicing (Third Filtering) ------------------------------------------------------------
    print("\n3. Dataset Slicing (Third Filtering) ===========================================================")
    print("Slicing datasets to 20% for faster processing...")
    train_hf_ds = train_hf_ds.shuffle(seed=SEED).select(range(min(len(train_hf_ds), int(len(train_hf_ds) * 0.2))))
    val_hf_ds = val_hf_ds.shuffle(seed=SEED).select(range(min(len(val_hf_ds), int(len(val_hf_ds) * 0.2))))
    test_hf_ds = test_hf_ds.shuffle(seed=SEED).select(range(min(len(test_hf_ds), int(len(test_hf_ds) * 0.2))))

    print(f"Train dataset size after 20% slicing: {len(train_hf_ds)}")
    print(f"Validation dataset size after 20% slicing: {len(val_hf_ds)}")
    print(f"Test dataset size after 20% slicing: {len(test_hf_ds)}")
    print("\n===============================================================================================\n")


    print("\nCreating model ==================================================================================")
    model = WhisperFlamingoModule(cfg, cfg.model_name, cfg.lang, 
                               train_hf_ds, 
                               val_hf_ds,
                               test_hf_ds)
    print("\n===============================================================================================\n")

    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    
    # Determine validation dataloaders
    # The original script used specific dataloaders for validation during fit and for trainer.validate()
    # Now val_dataloader() returns a list [noisy, clean]
    # test_dataloader() also returns [noisy, clean]
    
    # For trainer.fit, pass the list from val_dataloader
    fit_val_dataloaders = model.val_dataloader() 
    # For trainer.validate (before training), pass list from test_dataloader OR val_dataloader
    # Original script validated on test and val sets before training. Let's use test.
    pre_train_validate_dataloaders = model.test_dataloader()

    print("\nCreating Trainer ==================================================================================")
    trainer = Trainer(
        precision=getattr(cfg, 'precision', 16), # Default to 16
        strategy=strategy,
        accelerator="gpu", # Hardcoded, could be from cfg
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        num_sanity_val_steps=getattr(cfg, 'num_sanity_val_steps', 0),
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=getattr(cfg, 'reload_dataloaders_every_n_epochs', 1),
        use_distributed_sampler=False, # Handled by DistributedSamplerWrapper
        sync_batchnorm=getattr(cfg, 'sync_batchnorm', True),
        # MEMORY OPTIMIZATIONS
        enable_checkpointing=True
    )
    print("\n===============================================================================================\n")

    print("\nTraining ========================================================================================")
    print(f"Train ID: {cfg.train_id}")
    resume_ckpt_path = os.path.join(cfg.check_output_dir, cfg.train_id, "last.ckpt")
    
    if os.path.exists(resume_ckpt_path) and cfg.resume_training:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        # When resuming, trainer.fit will use the val_dataloaders provided to it.
        trainer.fit(model, ckpt_path=resume_ckpt_path, val_dataloaders=fit_val_dataloaders)
    else:
        if os.path.exists(resume_ckpt_path):
            print(f"Checkpoint found at {resume_ckpt_path} but resume_training is False. Starting new training.")
        else:
            print("No checkpoint found. Starting new training.")
        
        print("Validating before training...")
        try:
            trainer.validate(model=model, dataloaders=pre_train_validate_dataloaders)
        except Exception as e:
            print(f"Warning: Pre-training validation failed: {e}")
            print("Continuing with training anyway...")
        
        print("Starting training...")
        trainer.fit(model, val_dataloaders=fit_val_dataloaders)

    print("\nTraining finished! ==================================================================================")

    print("\nTesting =============================================================================================")
    print("Testing the best model...")
    # After training, test with the best checkpoint
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        print(f"Loading best model from: {best_ckpt_path}")
        # trainer.test(model, ckpt_path=best_ckpt_path, dataloaders=model.test_dataloader()) # This would re-init model
        # To test the already configured model object with best weights:
        model_best = WhisperFlamingoModule.load_from_checkpoint(best_ckpt_path, cfg=cfg, model_name=cfg.model_name, lang=cfg.lang,
                                                              train_hf_dataset=train_hf_ds, val_hf_dataset=val_hf_ds, test_hf_dataset=test_hf_ds)
        trainer.test(model=model_best, dataloaders=model_best.test_dataloader())
    else:
        print("Could not find the best model checkpoint to run final testing.")

    print(f"Script finished for train_id: {cfg.train_id}") 
    print("\n===============================================================================================\n")