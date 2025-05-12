import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import logging
from scipy.io import wavfile
from python_speech_features import logfbank
from datasets import DatasetDict
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from transformers import ProcessorMixin

#---------------------------------------------------------------------------------------------------------------------
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
#---------------------------------------------------------------------------------------------------------------------

#  Constants for data processing (can be moved to DataTrainingArguments or AV2TextConfig if preferred)
TARGET_AUDIO_SR = 16000
VIDEO_CROP_SIZE = 88
VIDEO_MEAN = 0.421
VIDEO_STD = 0.165

#=====================================================================================================================
# Video frame processing classes
#=====================================================================================================================

class Compose:
    """Compose several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class Normalize:
    """Normalize a numpy array with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        return (frames - self.mean) / self.std

class CenterCrop:
    """Center crop frames to the specified size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_h = (h - th) // 2
        delta_w = (w - tw) // 2
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames
#=======================================================================================================================================
def create_dataset_splits(dataset, test_size=0.2, val_size=0.1, train_size=0.7, dataset_name="ami", model_name="av_hubert"):
    """
    Create dataset splits from a HuggingFace dataset. 
    This will create a train-val-test split and save each of the splits to disk.
    """

    train_test_split = dataset.train_test_split(test_size=test_size) #80% train-val, 20% test
    test_split = train_test_split["test"]

    # Split the train-val dataset into train and validation
    train_val_split = train_test_split["train"].train_test_split(test_size=val_size) #70% train, 10% validation
    train_split = train_val_split["train"]
    val_split = train_val_split["test"]

    # Create a DatasetDict with the splits
    splits = DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": test_split
    })
    
    # Save the splits to disk
    train_split.save_to_disk(os.path.join("data", dataset_name, model_name, "train")) 
    val_split.save_to_disk(os.path.join("data", dataset_name, model_name, "validation"))
    test_split.save_to_disk(os.path.join("data", dataset_name, model_name, "test"))
    
    return splits
#---------------------------------------------------------------------------------------------------------------------

#=====================================================================================================================  
#                           Functions for Feature Processing
#=====================================================================================================================
            
def load_video_features(video_path, crop_size=VIDEO_CROP_SIZE, mean=VIDEO_MEAN, std=VIDEO_STD):
    """
    Load and process video frames
    """
    transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((crop_size, crop_size)),
        Normalize(mean, std)
    ])
    
    try:
        feats = load_video(video_path)
        feats = transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None
def load_audio_features(audio_path, target_sr=TARGET_AUDIO_SR):
    """
    Load audio file and extract features.
    Handles different sample rates and mono/stereo audio.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000)
        
    Returns:
        np.ndarray: Extracted audio features with shape [T, F]
    """
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # First try scipy.io.wavfile which is faster
            sample_rate, wav_data = wavfile.read(audio_path)
            
            # Handle stereo audio (convert to mono by averaging channels)
            if len(wav_data.shape) > 1 and wav_data.shape[1] > 1:
                logger.info(f"Converting stereo to mono for {audio_path}")
                wav_data = wav_data.mean(axis=1).astype(wav_data.dtype)
                
            # Resample if needed
            if sample_rate != target_sr:
                logger.info(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz for {audio_path}")
                # Use torchaudio for resampling
                wav_tensor = torch.tensor(wav_data).float()
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                wav_tensor = resampler(wav_tensor)
                wav_data = wav_tensor.numpy().astype(np.float32)
                sample_rate = target_sr
                
        except Exception as e:
            # Fallback to torchaudio which handles more formats
            logger.warning(f"Failed to load with scipy.io.wavfile, trying torchaudio: {e}")
            wav_tensor, sample_rate = torchaudio.load(audio_path)
            
            # Handle stereo (convert to mono by averaging channels)
            if wav_tensor.shape[0] > 1:
                logger.info(f"Converting stereo to mono for {audio_path}")
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != target_sr:
                logger.info(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz for {audio_path}")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                wav_tensor = resampler(wav_tensor)
                sample_rate = target_sr
                
            wav_data = wav_tensor.squeeze().numpy()

        # Normalize audio to [-1, 1] range
        if wav_data.dtype != np.float32:
            wav_data = wav_data.astype(np.float32)
            if wav_data.max() > 1.0:  # PCM data
                wav_data = wav_data / 32768.0  # 16-bit audio normalization
        
        # Extract log mel filterbank features
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
        audio_feats = audio_stacker(audio_feats, 4)  # [T/stack_order_audio, F*stack_order_audio]

        return audio_feats
    
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        return None
    

def audio_stacker(audio_feats, stack_order=4):
    """
    Concatenating consecutive audio frames
    """
    feat_dim = audio_feats.shape[1]
    if len(audio_feats) % stack_order != 0:
        res = stack_order - len(audio_feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(audio_feats.dtype)
        audio_feats = np.concatenate([audio_feats, res], axis=0)
    audio_feats = audio_feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return audio_feats

def load_video(video_path):
    """
    Load video file and preprocess frames.
    
    Args:
        video_path: Path to video file
        crop_size: Size to crop frames to (default: 88)
        mean: Mean for normalization (default: 0.421)
        std: Standard deviation for normalization (default: 0.165)
        
    Returns:
        torch.Tensor: Video frames tensor with shape [C, T, H, W]
    """
    # Stacking 4 frames together
    stack_order = 3
    for i in range(stack_order):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {video_path} ({i} / {stack_order})")
            if i == stack_order - 1:
                raise ValueError(f"Unable to load {video_path}")

#=====================================================================================================================
#                           Functions for AVHubert Dataset
# 
# - FIXME: FUNCTIONS BELOW NO LONGER USED
# - FIXME: Old approach, TRY new approach: using prepare_dataset() and DataCollator
#=====================================================================================================================
#---------------------------------------------------------------------------------------------------------------------

# @dataclass
# class AVHubertBatch:
#     """
#     Data class for storing batched inputs for the AVHubert model.
#     """
#     audio_values: Optional[torch.Tensor] = None  # [B, T_audio]
#     audio_attention_mask: Optional[torch.Tensor] = None  # [B, T_audio]
#     visual_values: Optional[torch.Tensor] = None  # [B, C, T_visual, H, W]
#     visual_attention_mask: Optional[torch.Tensor] = None  # [B, T_visual]
#     labels: Optional[torch.Tensor] = None  # [B, T_labels]
    
#     def to(self, device):
#         """Move the batch to device"""
#         for key, value in self.__dict__.items():
#             if isinstance(value, torch.Tensor):
#                 setattr(self, key, value.to(device))
#         return self


# def align_lengths(audio_feats, video_feats):
#     """
#     Align audio and video features to have the same temporal dimension.
#     For temporal alignment, we need to consider that:
#     - Audio is typically processed at a higher rate than video
#     - We need to make sure both have the same temporal dimension for fusion

#     Args:
#         audio_feats: Audio features tensor [B, T_audio, D_audio] (batch_size, time_steps, feature_dim)
#         video_feats: Video features tensor [B, T_video, D_video] (batch_size, time_steps, feature_dim)

#     Returns:
#         Tuple of aligned features tensors
#     """
#     if audio_feats is None or video_feats is None:
#         return audio_feats, video_feats
        
#     # Get temporal dimensions
#     B, T_audio, D_audio = audio_feats.shape
#     _, T_video, D_video = video_feats.shape
    
#     # If audio is longer than video, truncate audio
#     if T_audio > T_video:
#         audio_feats = audio_feats[:, :T_video, :]
#     # If video is longer than audio, truncate video
#     elif T_video > T_audio:
#         video_feats = video_feats[:, :T_audio, :]
        
#     return audio_feats, video_feats

# def process_data_for_avhubert(
#     audio_path=None, 
#     video_path=None, 
#     transcript=None,
#     processor=None,
#     max_audio_length=480000,  # 30 seconds at 16kHz
#     max_video_frames=750      # 30 seconds at 25 fps
# ):
#     """
#     Process audio and video data for the AVHubert model.
    
#     Args:
#         audio_path: Path to audio file
#         video_path: Path to video file
#         transcript: Transcript text for labels_ids
#         processor: HuggingFace processor for tokenization
#         max_audio_length: Maximum audio length in samples
#         max_video_frames: Maximum number of video frames
        
#     Returns:
#         Dict containing processed audio and video features
#     """
#     # Initialize with empty values
#     result = {
#         "audio_values": None,
#         "audio_attention_mask": None,
#         "visual_values": None,
#         "visual_attention_mask": None,
#         "labels": None
#     }
    
#     # Process audio if available
#     if audio_path is not None:
#         audio_values = load_audio(audio_path)
#         if audio_values is not None:
#             # Truncate or pad audio to max_length
#             if audio_values.shape[0] > max_audio_length:
#                 audio_values = audio_values[:max_audio_length]
#             elif audio_values.shape[0] < max_audio_length:
#                 padding = torch.zeros(max_audio_length - audio_values.shape[0])
#                 audio_values = torch.cat([audio_values, padding])
                
#             # Create attention mask (1 for real values, 0 for padding)
#             audio_attention_mask = torch.ones_like(audio_values)
#             if audio_values.shape[0] < max_audio_length:
#                 audio_attention_mask[-(max_audio_length - audio_values.shape[0]):] = 0
                
#             result["audio_values"] = audio_values
#             result["audio_attention_mask"] = audio_attention_mask
    
#     # Process video if available
#     if video_path is not None:
#         visual_values = load_video(video_path)
#         if visual_values is not None:
#             # visual_values is [C, T, H, W]
#             # Truncate or pad video frames
#             T = visual_values.shape[1]
#             if T > max_video_frames:
#                 visual_values = visual_values[:, :max_video_frames, :, :]
            
#             # Create attention mask (1 for real frames, 0 for padding)
#             visual_attention_mask = torch.ones(min(T, max_video_frames))
            
#             result["visual_values"] = visual_values
#             result["visual_attention_mask"] = visual_attention_mask
    
#     # Process transcript if available
#     if transcript is not None and processor is not None:
#         with processor.as_target_processor():
#             labels = processor(transcript).input_ids
#             result["labels"] = torch.tensor(labels)
    
#     return result

# def collate_audio_visual_batch(batch, processor=None):
#     """
#     Collate function for creating batches from multiple samples.
    
#     Args:
#         batch: List of dictionaries with audio_values, visual_values, etc.
#         processor: Optional processor for padding
        
#     Returns:
#         AVHubertBatch: Batch data for the model
#     """
#     # Initialize lists for each tensor type
#     audio_values = []
#     audio_attention_masks = []
#     visual_values = []
#     visual_attention_masks = []
#     labels = []
    
#     # Collect valid tensors from batch
#     has_audio = False
#     has_visual = False
#     has_labels = False
    
#     for item in batch:
#         if item.get("audio_values") is not None:
#             audio_values.append(item["audio_values"])
#             audio_attention_masks.append(item["audio_attention_mask"])
#             has_audio = True
            
#         if item.get("visual_values") is not None:
#             visual_values.append(item["visual_values"])
#             visual_attention_masks.append(item["visual_attention_mask"])
#             has_visual = True
            
#         if item.get("labels") is not None:
#             labels.append(item["labels"])
#             has_labels = True
    
#     # Pad and stack audio
#     if has_audio:
#         # Stack audio (assuming they're already padded to same length)
#         audio_values_tensor = torch.stack(audio_values)
#         audio_attention_mask_tensor = torch.stack(audio_attention_masks)
#     else:
#         audio_values_tensor = None
#         audio_attention_mask_tensor = None
    
#     # Pad and stack video
#     if has_visual:
#         # For video, we need to ensure all sequences have the same length in time dimension
#         max_frames = max(v.shape[1] for v in visual_values)
#         padded_visual_values = []
#         padded_visual_attention_masks = []
        
#         for i, video in enumerate(visual_values):
#             # Current shape: [C, T, H, W]
#             c, t, h, w = video.shape
#             if t < max_frames:
#                 padding = torch.zeros(c, max_frames - t, h, w)
#                 padded_video = torch.cat([video, padding], dim=1)
#                 # Update attention mask
#                 mask = visual_attention_masks[i]
#                 padding_mask = torch.zeros(max_frames - t)
#                 padded_mask = torch.cat([mask, padding_mask])
#                 padded_visual_values.append(padded_video)
#                 padded_visual_attention_masks.append(padded_mask)
#             else:
#                 padded_visual_values.append(video)
#                 padded_visual_attention_masks.append(visual_attention_masks[i])
        
#         visual_values_tensor = torch.stack(padded_visual_values)
#         visual_attention_mask_tensor = torch.stack(padded_visual_attention_masks)
#     else:
#         visual_values_tensor = None
#         visual_attention_mask_tensor = None
    
#     # Process labels if available
#     if has_labels and processor is not None:
#         # Use processor's padding for labels
#         with processor.as_target_processor():
#             labels_tensor = processor.pad(
#                 {"input_ids": labels},
#                 padding=True,
#                 return_tensors="pt",
#             ).input_ids
#     elif has_labels:
#         # Manual padding if no processor
#         max_len = max(len(l) for l in labels)
#         padded_labels = []
#         for l in labels:
#             padding = [-100] * (max_len - len(l))  # Use -100 as padding index
#             padded_labels.append(torch.tensor(list(l) + padding))
#         labels_tensor = torch.stack(padded_labels)
#     else:
#         labels_tensor = None
    
#     # Convert AVHubertBatch to a dictionary for Trainer compatibility
#     batch_dict = AVHubertBatch(
#         audio_values=audio_values_tensor,
#         audio_attention_mask=audio_attention_mask_tensor,
#         visual_values=visual_values_tensor,
#         visual_attention_mask=visual_attention_mask_tensor,
#         labels=labels_tensor
#     )
#     return batch_dict.__dict__ # Return as a dictionary

class AVHubertDataset(torch.utils.data.Dataset):
    """
    Dataset for loading and processing audio and visual data for AVHubert.
    
    This dataset handles loading audio and lip video data from a HuggingFace dataset,
    supporting modality dropping during training for robustness.
    """
    def __init__(
        self,
        dataset,
        processor=None,
        split="train",
        max_audio_length=480000,  # 30 seconds at 16kHz
        max_video_frames=750,     # 30 seconds at 25 fps
        audio_drop_prob=0.0,      # Probability of dropping audio during training
        visual_drop_prob=0.0,     # Probability of dropping visual during training
    ):
        self.dataset = dataset
        self.processor = processor
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_video_frames = max_video_frames
        self.audio_drop_prob = 0.0 if split != "train" else audio_drop_prob
        self.visual_drop_prob = 0.0 if split != "train" else visual_drop_prob
        
        # Log dataset info
        logger.info(f"Initializing AVHubertDataset (split={split}) with {len(dataset)} samples")
        logger.info(f"Using modality dropping in {split} mode: audio_drop_prob={self.audio_drop_prob}, visual_drop_prob={self.visual_drop_prob}")
        
        # Count available modalities
        self.count_available_modalities()
        
    def count_available_modalities(self):
        """Count samples with each modality available"""
        has_audio = 0
        has_video = 0
        has_both = 0
        
        for item in self.dataset:
            a = "audio" in item and item["audio"] is not None
            v = "lip_video" in item and item["lip_video"] is not None
            
            if a:
                has_audio += 1
            if v:
                has_video += 1
            if a and v:
                has_both += 1
                
        logger.info(f"Dataset modality statistics:")
        logger.info(f"  Total samples: {len(self.dataset)}")
        logger.info(f"  Samples with audio: {has_audio} ({has_audio/len(self.dataset)*100:.1f}%)")
        logger.info(f"  Samples with video: {has_video} ({has_video/len(self.dataset)*100:.1f}%)")
        logger.info(f"  Samples with both modalities: {has_both} ({has_both/len(self.dataset)*100:.1f}%)")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract paths and transcript correctly
        audio_item = item.get("audio", None)
        video_item = item.get("lip_video", None)
        transcript = item.get("transcript", None)

        audio_path = audio_item["path"] if isinstance(audio_item, dict) and "path" in audio_item else audio_item
        video_path = video_item["path"] if isinstance(video_item, dict) and "path" in video_item else video_item
        
        # Apply modality dropping for training
        if self.split == "train":
            if audio_path is not None and np.random.random() < self.audio_drop_prob:
                audio_path = None
            if video_path is not None and np.random.random() < self.visual_drop_prob:
                video_path = None
                
        # Ensure at least one modality is available
        if audio_path is None and video_path is None:
            # If both are None, set audio back (fallback)
            audio_item = item.get("audio", None)
            audio_path = audio_item["path"] if isinstance(audio_item, dict) and "path" in audio_item else audio_item
            if audio_path is None:
                # If still None, set video back
                video_item = item.get("lip_video", None)
                video_path = video_item["path"] if isinstance(video_item, dict) and "path" in video_item else video_item
        
        # Process the data
        result = process_data_for_avhubert(
            audio_path=audio_path,
            video_path=video_path,
            transcript=transcript,
            processor=self.processor,
            max_audio_length=self.max_audio_length,
            max_video_frames=self.max_video_frames
        )
        
        return result