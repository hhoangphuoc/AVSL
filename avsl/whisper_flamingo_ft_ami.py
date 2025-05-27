import os
import sys

# Add paths for whisper_flamingo and av_hubert ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo')
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert')

# Add to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)

# Ensure fairseq is properly accessible by adding the fairseq installation path
# This makes the fairseq modules visible without import conflicts
fairseq_path = os.path.join(av_hubert_path, 'fairseq')
if os.path.exists(fairseq_path) and fairseq_path not in sys.path:
    sys.path.insert(0, fairseq_path)
    print(f"✓ Added fairseq path to sys.path: {fairseq_path}")

# Set the correct av_hubert path for user module import
# This should point to the avhubert directory within the av_hubert repository
avhubert_user_dir = os.path.join(av_hubert_path, 'avhubert')
print(f"✓ AV-HuBERT user dir set to: {avhubert_user_dir}")

# Pre-import fairseq to ensure it's loaded correctly
try:
    import fairseq
    print(f"✓ Fairseq imported successfully from: {fairseq.__file__}")
    # Verify that the required modules are accessible
    _ = fairseq.checkpoint_utils
    _ = fairseq.utils
    print("✓ Fairseq checkpoint_utils and utils are accessible")
except Exception as e:
    print(f"Warning: Fairseq import issue: {e}")
    print("Continuing with the assumption that the original repository fix is in place...")


#===============================================================================================================
# FIXME: IS THIS NEEDED?
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
#===============================================================================================================

import yaml
import types
import numpy as np
import torch
from torch import nn
import cv2
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
)
from whisper_flamingo.utils_batch_samplers import LengthBatchSampler

SAMPLE_RATE = 16000 # Standard Whisper sample rate
AUDIO_HOP_LENGTH = 160 # For calculating number of frames from samples/duration
SEED = 3407
seed_everything(SEED, workers=True)


try:
    from whisper_flamingo.utils import load_video_feats
    from whisper_flamingo.spec_augment import spec_augment
except ImportError:
    print("Warning: Could not import helper functions from utils.py or spec_augment.py. Ensure they are in PYTHONPATH.")

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
        # if self.noise_files:
            # print(f"Loaded {len(self.noise_files)} noise files for augmentation.")

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
        
        audio = audio_data # No noise addition --------------------------------------------------------
        # Add noise (optional) - TODO: NOISE 
        # if self.train and self.noise_files and np.random.rand() < self.noise_prob:
            # add_noise expects int16 clean_wav if it's to scale with max_int16
            # For now, let's assume audio_data is float32 [-1, 1].
            # We might need to adapt add_noise or how we pass data to it.
            # For simplicity, let's assume add_noise can handle float32 input and output float32
            # Or, convert to int16, add noise, then back to float32.
            # temp_audio_int16 = (audio_data * 32767).astype(np.int16)
            # noisy_audio_int16 = add_noise(temp_audio_int16, self.noise_files, noise_snr=self.noise_snr)
            # audio_data = noisy_audio_int16.astype(np.float32) / 32768.0
            # Simpler: if add_noise can work with float audio directly (might need adjustment in utils.py)
            # For now, let's replicate Muavic's use of wavfile for consistency in add_noise if it relies on int16.
            # This part needs careful handling of dtypes for add_noise.
            # Muavic reads wav, gets int16, then add_noise, then float32.
            # To use `add_noise` as is, we'd need to simulate this.
            # However, let's assume for now a version of add_noise that can take float audio.
            # If not, this section needs refinement:
            # audio_data = add_noise_float(audio_data, self.noise_files, noise_snr=self.noise_snr) # Placeholder
            # For now, if noise is enabled, we'll just print a warning if using the dummy.
            # if callable(add_noise) and add_noise.__name__ != "_dummy_add_noise":
                 # Convert to int16 for add_noise as it's designed for wavfile output
                # temp_audio_int16 = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
                # noisy_audio_int16 = add_noise(temp_audio_int16, self.noise_files, noise_snr=self.noise_snr)
                # audio = noisy_audio_int16.astype(np.float32) / np.iinfo(np.int16).max
            # else:
                # audio = audio_data # No noise or dummy function
        # else:
            # audio = audio_data
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
        # Assuming item['lip_video'] is a path string. If it's raw data, load_video_feats needs adjustment.
        video_path_or_data = item['lip_video'] # NOTE: This is a HF Video() object

        if isinstance(video_path_or_data, dict) and 'path' in video_path_or_data:
             video_path = video_path_or_data['path']
        elif isinstance(video_path_or_data, str):
             video_path = video_path_or_data
        else:
            raise ValueError("Unsupported format for lip_video. Expected path string or dict with 'path'.")

        # load_video_feats returns T, H, W, C numpy array
        video_feats = load_video_feats(video_path, train=self.train) 
        video_feats = video_feats.astype(np.float32)

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

#===============================================================================================================
class WhisperFlamingoModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_hf_dataset, val_hf_dataset, test_hf_dataset) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        #===============================================================================================================
        # # Check if we have a local model file to load directly
        # local_model_path = getattr(cfg, 'local_whisper_model', None)
        # if local_model_path and os.path.exists(local_model_path):
        #     print(f"Loading Whisper model from local file: {local_model_path}")
        #     # Load the model from local checkpoint
        #     self.model = whisper.load_model(local_model_path,
        #                                     device='cpu',
        #                                     download_root=getattr(cfg, 'download_root', None),
        #                                     dropout_rate=getattr(cfg, 'dropout_rate', 0.0),
        #                                     video=True,
        #                                     video_model_path=getattr(cfg, 'video_model_ckpt', None), 
        #                                     prob_av=getattr(cfg, 'prob_use_av', 0.5), 
        #                                     prob_a=getattr(cfg, 'prob_use_a', 0.5),
        #                                     av_hubert_encoder=getattr(cfg, 'use_av_hubert_encoder', False),
        #                                     av_fusion=getattr(cfg, 'av_fusion', 'early'),
        #                                     add_gated_x_attn=getattr(cfg, 'add_gated_x_attn', 0))
        # else:
            # Try to load from download_root or download if necessary
        #===============================================================================================================
        print("Loading Whisper model from download_root or download if necessary")
        try:
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
            print("✓ Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            print("Please download the model manually or check your internet connection.")
            raise e
        #===============================================================================================================
        
        # Load pre-trained checkpoint if specified
        if cfg.pt_ckpt != '':
            state_dict = torch.load(cfg.pt_ckpt, map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v  for k, v in state_dict.items()}
            print(f"Loading pre-trained audio weights from: {cfg.pt_ckpt}")
            try:
                self.model.load_state_dict(state_dict_updated) 
            except BaseException as e: 
                print(str(e))
                print("Loading weights with strict=False")
                self.model.load_state_dict(state_dict_updated, strict=False) 
        
        self.freeze_video_model = cfg.freeze_video_model
        self.freeze_video_batch_norm_stats = cfg.freeze_video_batch_norm_stats
        
        multilingual = True if 'large' in model_name or '.en' not in model_name else False
        print(f"Multilingual tokenizer : {multilingual}, Language for dataset: {lang}")

        # --- Use standard Whisper tokenizer (skip custom token for now) ------------------------------------------------------------
        # For now, we'll use the standard Whisper tokenizer without custom tokens
        # Custom tokens can be added later if needed for specific use cases
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=multilingual, language=lang, task='transcribe'
        )
        print(f"Using standard Whisper tokenizer with vocab size: {self.tokenizer.encoding.n_vocab}")
        # --- End of tokenizer setup ------------------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # Whisper's default ignore_index for padding

        self.cfg = cfg
        self.lang = lang # Store lang for dataset
        
        # Store HF datasets
        self.train_hf_dataset = train_hf_dataset
        self.val_hf_dataset = val_hf_dataset
        self.test_hf_dataset = test_hf_dataset
        
        # Pre-calculate audio lengths for LengthBatchSampler
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
        
        features, x_v = self.model.encoder(input_ids, video, training=True, padding_mask=padding_mask)
        out = self.model.decoder(dec_input_ids, features, xv=x_v)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
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
                # Decode, removing all special tokens for WER/CER
                o_list.append(self.tokenizer.decode([t for t in o_tokens if t.item() not in self.special_token_set]))
                l_list.append(self.tokenizer.decode([t for t in l_tokens if t.item() not in self.special_token_set]))
            
            wer, cer = wer_cer(hypo=o_list, ref=l_list)
        
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
        # Prepare noise_config for AmiVideoHFDataset - REMOVED
        if train_mode:
            # noise_conf = {
            #     "prob": self.cfg.noise_prob,
            #     "noise_files": [self.cfg.noise_fn] if self.cfg.noise_fn else [],
            #     "snr": self.cfg.noise_snr_train
            # }
            spec_aug_conf = self.cfg.spec_augment
        else: 
            # Validation/Test
            # noise_fn_key = 'noise_fn_val' if 'val' in hf_dataset.split else 'noise_fn_test'
            # current_noise_fn = getattr(self.cfg, noise_fn_key, None)
            
            # noise_conf = {
            #     "prob": noise_override_prob if noise_override_prob is not None else 0, # Controlled by specific dataloader
            #     "noise_files": [current_noise_fn] if current_noise_fn else [],
            #     "snr": self.cfg.noise_snr_eval if hasattr(self.cfg, 'noise_snr_eval') else self.cfg.noise_snr_train # Default to train SNR if eval SNR not set
            # }
            spec_aug_conf = False # No spec augment for val/test

        dataset = AmiVideoHFDataset(hf_dataset, 
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    audio_max_length=self.dataset_audio_max_length, # For pad_or_trim
                                    lang_code=self.lang,
                                    spec_augment_config=spec_aug_conf,
                                    # noise_config=None, # Noise config removed
                                    train=train_mode)
        
        # cfg.audio_max_length is for batch sampler (max samples per batch item, not total samples in batch)
        # cfg.batch_size is items per batch
        # batch_bins = total_samples_in_batch = cfg.audio_max_length * cfg.batch_size
        batch_bins = int(self.cfg.audio_max_length * (self.cfg.batch_size if train_mode else self.cfg.eval_batch_size if hasattr(self.cfg, 'eval_batch_size') else self.cfg.batch_size * 4)) # Larger batch for eval
        
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
                                           pin_memory=True) # Added pin_memory

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_flamingo_ft_ami.py <config_yaml_path>")
        sys.exit(1)
        
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print("Configuration:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    
    print(f"Audio max length for batch sampler: {cfg.audio_max_length} samples")
    dataset_audio_max_len_cfg = getattr(cfg, 'dataset_audio_max_length', whisper.audio.N_SAMPLES)
    print(f"Audio max length for dataset (pad_or_trim): {dataset_audio_max_len_cfg} samples")


    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor)
    
    # Load Hugging Face datasets
    print(f"Loading train dataset from: {cfg.train_data_path}")
    train_hf_ds = load_from_disk(cfg.train_data_path)
    print(f"Loading validation dataset from: {cfg.val_data_path}")
    val_hf_ds = load_from_disk(cfg.val_data_path)
    print(f"Loading test dataset from: {cfg.test_data_path}")
    test_hf_ds = load_from_disk(cfg.test_data_path)

    # Filter datasets by audio length if specified in config (e.g. cfg.max_duration_seconds)
    # This is to avoid very long samples that might cause OOM with fixed batching,
    # LengthBatchSampler helps but extreme outliers can still be an issue.
    # whisper.audio.N_SAMPLES is 30s. cfg.audio_max_length is for batcher in samples.
    # Max duration for filtering should be based on what the model can handle or what's practical.
    max_duration_filter_sec = getattr(cfg, 'max_duration_filter_seconds', 30.0) # Max 30s by default
    
    # train_hf_ds = train_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    # val_hf_ds = val_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    # test_hf_ds = test_hf_ds.filter(lambda x: x['duration'] <= max_duration_filter_sec)
    
    print(f"Train dataset size : {len(train_hf_ds)}")
    print(f"Validation dataset : {len(val_hf_ds)}")
    print(f"Test dataset size : {len(test_hf_ds)}")

    # --- ONLY USE 10% OF DATA FOR TRAINING ------------------------------------------------------------
    print("Slicing datasets to 10% for faster processing...")
    train_hf_ds = train_hf_ds.shuffle(seed=SEED).select(range(min(len(train_hf_ds), int(len(train_hf_ds) * 0.1))))
    val_hf_ds = val_hf_ds.shuffle(seed=SEED).select(range(min(len(val_hf_ds), int(len(val_hf_ds) * 0.1))))
    test_hf_ds = test_hf_ds.shuffle(seed=SEED).select(range(min(len(test_hf_ds), int(len(test_hf_ds) * 0.1))))

    print(f"Train dataset size after 10% slicing: {len(train_hf_ds)}")
    print(f"Validation dataset size after 10% slicing: {len(val_hf_ds)}")
    print(f"Test dataset size after 10% slicing: {len(test_hf_ds)}")
    # --- End of 10% slicing ------------------------------------------------------------


    model = WhisperFlamingoModule(cfg, cfg.model_name, cfg.lang, 
                               train_hf_ds, 
                               val_hf_ds,
                               test_hf_ds)
    
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
    )

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
        trainer.validate(model=model, dataloaders=pre_train_validate_dataloaders)
        
        print("Starting training...")
        trainer.fit(model, val_dataloaders=fit_val_dataloaders)

    print("Training finished.")
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