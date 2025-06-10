import os
import sys
import jiwer
# from transformers import WhisperTokenizer

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
    print("Warning: Could not import memory_utils")

# AGGRESSIVE MEMORY OPTIMIZATION SETTINGS =====================================
# Force strict memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDNN_DISABLE'] = '0'

# Limit video frame processing to reduce memory
MAX_VIDEO_FRAMES = 25  # Limit to ~1 second at 25fps
VIDEO_DOWNSAMPLE_FACTOR = 2  # Process every 2nd frame
# =============================================================================

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
        # debug_hf_video_object,
        # create_robust_video_filter,
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


# Set environment variable to allow expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
import jiwer

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
        
        # Normalize text for the custom <laugh> token
        text = text.lower()
        text = text.replace('[laugh]', '<laugh>').replace('(laugh)', '<laugh>')
        text = text.replace('<laughter>', '<laugh>').replace('(laughter)', '<laugh>')

        # preprocess text
        text_transformation = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.SubstituteWords({
                "'cause": "because",
                "cuz": "because",
                "c'mon": "come on"
            }),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ToLowerCase()
        ])
        text = text_transformation(text)

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

        max_video_len_processed_audio = round(len(audio) / self.target_sample_rate * 25)

        # AGGRESSIVE VIDEO MEMORY OPTIMIZATION =====================================
        # Limit video frames to prevent memory explosion
        max_video_frames = min(MAX_VIDEO_FRAMES, max_video_len_processed_audio)
        
        if len(video_feats) > max_video_frames:
            # Downsample video frames if too long
            step = max(1, len(video_feats) // max_video_frames)
            video_feats = video_feats[::step][:max_video_frames]
        
        # Further limit if still too long
        if len(video_feats) > max_video_frames:
            video_feats = video_feats[:max_video_frames]
        
        # Compress video features to save memory (reduce precision)
        if self.train:
            # Use lower precision for training to save memory
            video_feats = video_feats.astype(np.float16).astype(np.float32)
        # =====================================================================
        
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
            multilingual=multilingual, language=lang, task='transcribe')

        # --- Use Hugging Face tokenizer to allow adding custom tokens ---------------------------------------
        # NOTE: TRY THIS LATER IF IT WORKS
        # print("✓ Loading Hugging Face WhisperTokenizer to allow for custom tokens...")
        # self.tokenizer = WhisperTokenizer.from_pretrained(
        #     cfg.model_name, language=lang, task="transcribe"
        # )
        
        # # Add custom token for laughter
        # print("✓ Adding custom token <laugh> to tokenizer")
        # new_tokens = ['<laugh>']
        # self.tokenizer.add_tokens(new_tokens)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        # laugh_token_id = self.tokenizer.convert_tokens_to_ids('<laugh>')
        # print(f"✓ The token ID for '<laugh>' is: {laugh_token_id}")

        # print(f"\n✓ Using Hugging Face tokenizer with vocab size: {self.tokenizer.vocab_size}\n")
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
        
        # AGGRESSIVE MEMORY MANAGEMENT ==========================================
        # Clear cache more frequently for memory-constrained training
        if batch_id % 5 == 0:  # Clear cache every 5 batches instead of 10
            try:
                clear_gpu_memory()
            except NameError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Move data to GPU only when needed and clear immediately after use
        try:
            features, x_v = self.model.encoder(input_ids, video, training=True, padding_mask=padding_mask)
            out = self.model.decoder(dec_input_ids, features, xv=x_v)
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            
            # Clear intermediate tensors immediately
            del features, x_v, out
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error in training step {batch_id}. Clearing cache and retrying...")
                # Clear all cached memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise e
            else:
                raise e
        # =====================================================================
        
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
        # AGGRESSIVE VALIDATION MEMORY MANAGEMENT ===============================
        # Process validation with minimal memory footprint
        try:
            video = batch["video"]
            input_ids = batch["input_ids"]
            labels = batch["labels"].long()
            dec_input_ids = batch["dec_input_ids"].long()
            padding_mask = batch["padding_mask"]

            # Clear cache before validation step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Only process AV modality as per config (add_gated_x_attn=1)
            mod_list = {"av": None}
            
            with torch.no_grad():
                # Process with gradient checkpointing disabled for inference
                features_av, x_v = self.model.encoder(input_ids, video, track_norm=False, padding_mask=padding_mask)
                out_av = self.model.decoder(dec_input_ids, features_av, xv=x_v)
                loss = self.loss_fn(out_av.view(-1, out_av.size(-1)), labels.view(-1))
            
            tokens = torch.argmax(out_av, dim=2)
            
            # Immediately move results to CPU and clear GPU tensors
            tokens_cpu = tokens.cpu()
            labels_cpu = labels.cpu()
            loss_cpu = loss.cpu()
            
            # Aggressively delete GPU tensors
            del video, input_ids, labels, dec_input_ids, padding_mask, features_av, x_v, out_av, tokens, loss
            
            # Force memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM in validation step {batch_id}. Skipping this batch.")
                # Return dummy values to continue training
                return {"val/loss_av": 0.0, "val/wer_av": 1.0, "val/cer_av": 1.0}
            else:
                raise e
        # =====================================================================
        
        # Masking tokens after EOT on CPU
        eot_find = (torch.where(tokens_cpu == self.tokenizer.eot, 1, 0))
        if eot_find.shape[1] > 0:
            first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1, device=tokens_cpu.device) * eot_find, dim=1, keepdim=True)
            token_indices = torch.arange(eot_find.shape[1], device=tokens_cpu.device)
            tokens_cpu[token_indices.unsqueeze(0) > first_eot] = self.tokenizer.eot

        # Decode on CPU
        o_list, l_list = [], []
        for o_tokens, l_tokens in zip(tokens_cpu, labels_cpu):
            valid_o_tokens = [t for t in o_tokens if t.item() >= 0 and t.item() not in self.special_token_set]
            valid_l_tokens = [t for t in l_tokens if t.item() >= 0 and t.item() not in self.special_token_set]
            o_list.append(self.tokenizer.decode(valid_o_tokens))
            l_list.append(self.tokenizer.decode(valid_l_tokens))

        # --- Text Normalization for WER/CER ---
        transformation = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemovePunctuation(),
            jiwer.SubstituteWords({
                "'cause": "because",
                "cuz": "because",
                "c'mon": "come on"
            }),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ToLowerCase()
        ])
        
        norm_o_list = []
        norm_l_list = []
        for ref, hypo in zip(l_list, o_list):
            if ref.strip() or hypo.strip(): # Process if at least one is not empty
                norm_l_list.append(transformation(ref))
                norm_o_list.append(transformation(hypo))

        # Calculate WER and CER on normalized text
        wer, cer = wer_cer(hypo=norm_o_list, ref=norm_l_list)
        
        if batch_id < 2:
            print(f"Mod: av, Dataloader Idx: {dataloader_idx}")
            for i, (hypo, ref) in enumerate(zip(norm_o_list, norm_l_list)): # Print normalized versions
                print("-" * 10)
                print(f"PRED ({i}): {hypo}")
                print(f"REF  ({i}): {ref}")
                if i == 1: break
        
        log_prefix_map = {0: 'test', 1: 'val'}
        log_prefix = log_prefix_map.get(dataloader_idx, f'unknown_idx_{dataloader_idx}')
        
        self.log(f"{log_prefix}/loss_av", loss_cpu, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{log_prefix}/cer_av", cer, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"{log_prefix}/wer_av", wer, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Clear CUDA cache after each validation batch to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
        spec_aug_conf = self.cfg.spec_augment if train_mode else False

        dataset = AmiVideoHFDataset(hf_dataset, 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    audio_max_length=self.dataset_audio_max_length,
                                    lang_code=self.lang,
                                    spec_augment_config=spec_aug_conf,
                                    train=train_mode)
        
        batch_size = self.cfg.batch_size if train_mode else self.cfg.eval_batch_size
        batch_bins = int(self.cfg.audio_max_length * batch_size)
        print(f"Using {'training' if train_mode else 'evaluation'} batch size: {batch_size}")
        
        length_sorter = LengthBatchSampler(batch_bins=batch_bins,
                                           shapes=audio_lengths,
                                           sort_in_batch='descending',
                                           sort_batch='shuffle' if train_mode else 'descending',
                                           drop_last=train_mode)
        
        if self.cfg.num_devices > 1:
            print("Using distributed sampler for DDP.")
            length_sorter = DistributedSamplerWrapper(length_sorter, shuffle=train_mode)

        # MEMORY OPTIMIZED DATALOADER SETTINGS ================================
        num_workers = getattr(self.cfg, 'num_worker', 0)
        pin_memory = getattr(self.cfg, 'pin_memory', False) 
        persistent_workers = getattr(self.cfg, 'persistent_workers', False)
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=length_sorter,
                                           num_workers=num_workers,
                                           collate_fn=WhisperVideoCollatorWithPadding(),
                                           pin_memory=pin_memory,
                                           persistent_workers=persistent_workers)
        # =====================================================================

    def train_dataloader(self):
        return self._create_dataloader(self.train_hf_dataset, self.train_audio_lengths, train_mode=True)

    def val_dataloader(self):
        val_clean_loader = self._create_dataloader(self.val_hf_dataset, self.val_audio_lengths, train_mode=False)
        return [val_clean_loader] 

    def test_dataloader(self):
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
    print(f"Loading configuration from: {cfg_yaml}")
    
    if not os.path.exists(cfg_yaml):
        print(f"ERROR: Configuration file not found: {cfg_yaml}")
        sys.exit(1)
        
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)
    
    print(f"✅ Configuration loaded successfully")

    print("\nConfiguration ================================================================")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print("\n============================================================================\n")

    # Early checks for essential dataset paths from config--------------------------------------------------------------------------------------------------------
    essential_paths_from_config = {
        "train_data_path": getattr(cfg, 'train_data_path', None),
        "val_data_path": getattr(cfg, 'val_data_path', None),
        "test_data_path": getattr(cfg, 'test_data_path', None),
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


    # Ensure output directories exist
    os.makedirs(cfg.log_output_dir, exist_ok=True)
    os.makedirs(cfg.check_output_dir, exist_ok=True)
    print(f"✅ Created/verified output directories:")
    print(f"  Log output: {cfg.log_output_dir}")
    print(f"  Checkpoint output: {cfg.check_output_dir}")
    
    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor)
    
    # Load Hugging Face datasets
    print("\nLoading datasets ============================================================")
    
    #FIXME: IS THIS NEEDED?
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
    
    # First try: Load validated datasets
    datasets_loaded = False
    try:
        print("🔍 TRY: Loading validated videos (already processed)...")

        paths_valid = (
            verify_path_exists(cfg.train_data_path, "Train") and
            verify_path_exists(cfg.val_data_path, "Validation") and
            verify_path_exists(cfg.test_data_path, "Test")
        )
        
        if not paths_valid:
            print("❌ One or more validated dataset paths do not exist. Trying fallback approach.")
            raise FileNotFoundError("One or more validated dataset paths do not exist")
        
        print(f"Loading validated train dataset from: {cfg.train_data_path}")
        train_hf_ds = load_from_disk(cfg.train_data_path)
        print(f"✅ Validated train dataset loaded: {len(train_hf_ds)} samples")
        
        print(f"Loading validated validation dataset from: {cfg.val_data_path}")
        val_hf_ds = load_from_disk(cfg.val_data_path)
        print(f"✅ Validated validation dataset loaded: {len(val_hf_ds)} samples")
        
        print(f"Loading validated test dataset from: {cfg.test_data_path}")
        test_hf_ds = load_from_disk(cfg.test_data_path)
        print(f"✅ Validated test dataset loaded: {len(test_hf_ds)} samples")
        
        debug_dataset_sample(train_hf_ds, "Train")
        
        datasets_loaded = True
    
    except FileNotFoundError as e:
        print(f"❌ Path error loading validated datasets: {e}")
    
    except ImportError as e:
        print(f"❌ Import error loading datasets: {e}")
        print("This may be due to a version mismatch in the datasets library.")
        sys.exit(1)  # Critical error, cannot continue
        
    except ValueError as e:
        print(f"❌ Value error loading datasets: {e}")
        print("This may be due to corrupted dataset files.")
        
    except Exception as e:
        print(f"❌ Unexpected error loading validated datasets: {e}")
        print(f"Error type: {type(e).__name__}")
    
    # Final verification
    if not datasets_loaded or len(train_hf_ds) == 0:
        print("❌ FATAL ERROR: Failed to load datasets or train dataset is empty")
        sys.exit(1)
    print("\n===============================================================================================\n")

    print(f"Train dataset size: {len(train_hf_ds)}")
    print(f"Validation dataset size: {len(val_hf_ds)}")
    print(f"Test dataset size: {len(test_hf_ds)}")
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
    print("Slicing datasets to 5% for memory-constrained training...")
    # # Further reduce dataset size for memory constraints
    # train_slice_ratio = 0.05  # 5% of data
    # val_slice_ratio = 0.1    # 10% of validation data
    # test_slice_ratio = 0.1   # 10% of test data
    
    # train_hf_ds = train_hf_ds.shuffle(seed=SEED).select(range(min(len(train_hf_ds), int(len(train_hf_ds) * train_slice_ratio))))
    # val_hf_ds = val_hf_ds.shuffle(seed=SEED).select(range(min(len(val_hf_ds), int(len(val_hf_ds) * val_slice_ratio))))
    # test_hf_ds = test_hf_ds.shuffle(seed=SEED).select(range(min(len(test_hf_ds), int(len(test_hf_ds) * test_slice_ratio))))

    # print(f"Train dataset size after {train_slice_ratio*100}% slicing: {len(train_hf_ds)}")
    # print(f"Validation dataset size after {val_slice_ratio*100}% slicing: {len(val_hf_ds)}")
    # print(f"Test dataset size after {test_slice_ratio*100}% slicing: {len(test_hf_ds)}")
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
    
    # GPU Detection and Configuration
    print("🔍 GPU Detection and Configuration: ================================================================")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    # Check environment variables for GPU allocation
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    slurm_gpus = os.environ.get('SLURM_GPUS', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"SLURM_GPUS: {slurm_gpus}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Test basic GPU operations
        try:
            test_tensor = torch.randn(2, 2).cuda()
            print("✅ GPU tensor creation successful")
            accelerator_to_use = getattr(cfg, 'accelerator', 'gpu')
            devices_to_use = getattr(cfg, 'num_devices', 1)
        except Exception as e:
            print(f"❌ GPU tensor creation failed: {e}")
            print("⚠️ Falling back to CPU despite CUDA being available")
            accelerator_to_use = 'cpu'
            devices_to_use = 1
    else:
        print("⚠️ CUDA not available, falling back to CPU")
        print("Note: Training on CPU will be extremely slow for this model size")
        accelerator_to_use = 'cpu'
        devices_to_use = 1
    
    print(f"Using accelerator: {accelerator_to_use}, devices: {devices_to_use}")
    print("\n===============================================================================================\n")
    
    trainer = Trainer(
        precision=getattr(cfg, 'precision', 'bf16'), # Use bfloat16 for better memory efficiency
        strategy=strategy,
        accelerator=accelerator_to_use,
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        num_sanity_val_steps=getattr(cfg, 'num_sanity_val_steps', 0),
        devices=devices_to_use,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=getattr(cfg, 'reload_dataloaders_every_n_epochs', 0),
        use_distributed_sampler=False,
        sync_batchnorm=getattr(cfg, 'sync_batchnorm', False),
        # AGGRESSIVE MEMORY OPTIMIZATIONS =====================================
        enable_checkpointing=True,
        limit_train_batches=0.1,  # Use only 10% of training data per epoch
        limit_val_batches=0.05,   # Use only 5% of validation data
        # ===================================================================
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