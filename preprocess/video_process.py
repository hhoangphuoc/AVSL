"""
This script contains function to process video source input, including: 
    - segmenting video based on transcript_segments timestamps, video is resampled to 25fps, mp4 format
    - extracting lip-reading region of video with 96
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ffmpeg
import cv2
import numpy as np
from torchvision import transforms
from collections import deque
import torch

from utils import (
    create_dlib_detectors, 
    detect_landmarks, 
    landmarks_interpolate, 
    warp_img, 
    apply_transform, 
    cut_patch
)
from preprocess.constants import CNN_DETECTOR_PATH, FACE_PREDICTOR_PATH, MEAN_FACE_PATH

def segment_video(video_file, start_time, end_time, video_output_file):
    """
    Segment a video file between start_time and end_time, and save it to `video_output_file`.
    Video is resampled to 25fps and output format is mp4.
    
    Args:
        video_file: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        video_output_file: Path to save the output video (.mp4)
    
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Check that output file ends with .mp4
        if not video_output_file.lower().endswith('.mp4'):
            original_output = video_output_file
            video_output_file = f"{os.path.splitext(video_output_file)[0]}.mp4"
            print(f"Changing output format from {original_output} to {video_output_file}")
            
        # Check for valid timestamps
        if start_time < 0:
            print(f"Warning: Negative start time {start_time} for {video_file}, setting to 0")
            start_time = 0
            
        # Get video duration to validate end_time - FIXME: Remove this part---------------
        probe = ffmpeg.probe(video_file)
        video_duration = float(probe['format']['duration'])
        # ---------------------------------------------------------------------------------
        
        if end_time > video_duration:
            print(f"Warning: End time {end_time} exceeds video duration {video_duration} for {video_file}, truncating")
            end_time = video_duration
            
        if start_time >= end_time:
            print(f"Warning: Invalid video segment {start_time}-{end_time} for {video_file}")
            return False, None
            
        # Calculate duration
        duration = end_time - start_time
        
        # Use ffmpeg to segment the video with 25fps
        (
            ffmpeg
            .input(video_file, ss=start_time, t=duration)
            .output(
                video_output_file, 
                r=25,                  # Set frame rate to 25fps
                vcodec='libx264',      # Video codec
                acodec='aac',          # Audio codec
                format='mp4',          # Ensure mp4 format
                **{'copyts': None}     # Preserve timestamps for better alignment
            )
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Verify the output video
        if os.path.exists(video_output_file) and os.path.getsize(video_output_file) > 0:
            return True, video_output_file
        else:
            print(f"Error: Output video file {video_output_file} is empty or doesn't exist")
            return False, None
            
    except ffmpeg.Error as e:
        print(f"Error segmenting video {video_file}: {e.stderr}")
        return False, None
    except Exception as e:
        print(f"Unexpected error processing video {video_file}: {str(e)}")
        return False, None

def load_video(video_path):
    """
    Load a video file from `video_path` and extract all frames as grayscale. 
    This is used for capturing sequence of frames from video and used for lip-reading.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        numpy.ndarray: Array of grayscale frames with shape [T, H, W]
    """
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
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
            
        frames = np.stack(frames)
        return frames
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        raise e



def extract_lip_frames(video_path, 
                       face_predictor_path=None, 
                       cnn_detector_path=None, 
                       mean_face_path=None, 
                       width_roi=96, 
                       height_roi=96, 
                       start_idx=48, 
                       stop_idx=68):
    """
    Extract lip regions from a video file.
    
    Args:
        video_path: Path to the video file
        face_predictor_path: Path to the face predictor model file.
        cnn_detector_path: Path to the CNN face detector model file. 
        mean_face_path: Path to the mean face reference file.
        width_roi: Width of the lip crop ROI
        height_roi: Height of the lip crop ROI
        start_idx: Start index of mouth landmarks
        stop_idx: End index of mouth landmarks
        
    Returns:
        numpy.ndarray: Array of lip frames with shape [T, H, W]
    """
    # Load mean face
    try:
        mean_face_landmarks = np.load(mean_face_path)
    except Exception as e:
        print(f"Error loading mean face landmarks from {mean_face_path}: {str(e)}")
        raise
    
    # Create detectors
    try:
        detector, cnn_detector, predictor = create_dlib_detectors(face_predictor_path, cnn_detector_path)
    except Exception as e:
        print(f"Error creating face detectors: {str(e)}")
        raise
    
    # Load video frames
    frames = load_video(video_path)
    
    # Detect landmarks for each frame
    landmarks = []
    for frame in frames:
        landmark = detect_landmarks(frame, detector, cnn_detector, predictor)
        landmarks.append(landmark)
    
    # Interpolate landmarks where detection failed
    landmarks = landmarks_interpolate(landmarks)
    if landmarks is None:
        raise Exception(f"No face detected in video {video_path}")
    
    # Define stable points for alignment
    stablePntsIDs = [33, 36, 39, 42, 45]
    STD_SIZE = (300, 300)
    window_margin = 12
    
    # Process frames to extract lip regions
    frame_idx = 0
    sequence = []
    margin = min(len(frames), window_margin)
    q_frame, q_landmarks = deque(), deque()
    
    for frame in frames:
        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            
            # Apply affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                        mean_face_landmarks[stablePntsIDs, :],
                                        cur_frame,
                                        STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            
            # Crop mouth patch
            try:
                sequence.append(cut_patch(trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        height_roi//2,
                                        width_roi//2))
            except Exception as e:
                print(f"Error cropping frame {frame_idx}: {str(e)}")
                # If cropping fails, add a black frame to maintain synchronization
                sequence.append(np.zeros((height_roi, width_roi), dtype=np.uint8))
        
        frame_idx += 1
    
    # Process remaining frames
    while q_frame:
        cur_frame = q_frame.popleft()
        # Transform frame
        trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
        # Transform landmarks
        trans_landmarks = trans(q_landmarks.popleft())
        # Crop mouth patch
        try:
            sequence.append(cut_patch(trans_frame,
                                    trans_landmarks[start_idx:stop_idx],
                                    height_roi//2,
                                    width_roi//2))
        except Exception as e:
            print(f"Error cropping frame: {str(e)}")
            sequence.append(np.zeros((height_roi, width_roi), dtype=np.uint8))
    
    return np.array(sequence)

def video_to_tensor(frames, normalize=True, image_mean=0.0, image_std=1.0, image_crop_size=88):
    """
    Convert video frames to tensor format suitable for AV-HuBERT.
    
    Args:
        frames: numpy.ndarray of shape [T, H, W]
        normalize: Whether to normalize pixel values
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        image_crop_size: Size to crop the frames to
        
    Returns:
        numpy.ndarray: Processed frames with shape [T, H, W, 1]
    """
    # Define transforms
    transform = transforms.Compose([
        lambda x: x.astype(np.float32),
        lambda x: (x - 0.0) / 255.0,  # Normalize to [0, 1]
        lambda x: transforms.functional.center_crop(torch.from_numpy(x), (image_crop_size, image_crop_size)).numpy(),
        lambda x: (x - image_mean) / image_std if normalize else x,
        lambda x: np.expand_dims(x, axis=-1)  # Add channel dimension [T, H, W, 1]
    ])
    
    # Apply transforms
    return transform(frames)

def process_video_for_av_hubert(video_path, 
                                face_predictor_path=None, 
                                cnn_detector_path=None, 
                                mean_face_path=None,
                                normalize=True, 
                                image_mean=0.0, 
                                image_std=1.0, 
                                image_crop_size=88):
    """
    Process a HuggingFace Video object for AV-HuBERT Video Encoder.
    
    Args:
        video_path: Path to the video file
        face_predictor_path: Path to the face predictor model file
        cnn_detector_path: Path to the CNN face detector model file
        mean_face_path: Path to the mean face reference file
        normalize: Whether to normalize pixel values
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        image_crop_size: Size to crop the frames to
        
    Returns:
        numpy.ndarray: Processed video features with shape [T, H, W, 1]
    """
    
    try:
        
        # Extract lip frames
        lip_frames = extract_lip_frames(
            video_path, 
            face_predictor_path, 
            cnn_detector_path, 
            mean_face_path
        )
        
        # Convert to tensor format
        video_features = video_to_tensor(
            lip_frames,
            normalize=normalize,
            image_mean=image_mean,
            image_std=image_std,
            image_crop_size=image_crop_size
        )
        
        return video_features

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        raise e

def save_lip_frames_to_video(lip_frames, output_path, fps=25, method='opencv'):
    """
    Save lip-cropped frames as an MP4 video.
    
    Args:
        lip_frames: numpy.ndarray of shape [T, H, W] containing lip-cropped frames
        output_path: Path to save the output video (.mp4)
        fps: Frame rate of the output video, defaults to 25fps
            method: Method to use for output video ('opencv' or 'torch')
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Check that output file ends with .mp4
        if not output_path.lower().endswith('.mp4'):
            original_output = output_path
            output_path = f"{os.path.splitext(output_path)[0]}.mp4"
            print(f"Changing output format from {original_output} to {output_path}")
        
        if method == 'opencv':
            # Method 1: Using OpenCV's VideoWriter
            h, w = lip_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Convert grayscale to RGB if needed
            for frame in lip_frames:
                if len(frame.shape) == 2:  # grayscale
                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame_rgb = frame
                out.write(frame_rgb)
            
            out.release()

        elif method == 'torch':
            # Method 3: Using torchvision
            import torchvision
            
            # Convert frames to proper format for torchvision
            frames_tensor = []
            for frame in lip_frames:
                if len(frame.shape) == 2:  # grayscale
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                # Convert to torch tensor [C, H, W]
                frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1))
                frames_tensor.append(frame_tensor)
            
            # Stack all frames into a single tensor [T, C, H, W]
            video_tensor = torch.stack(frames_tensor)
            
            # Write video
            torchvision.io.write_video(
                output_path,
                video_tensor,
                fps=fps,
                video_codec="libx264",
                options={"crf": "23", "preset": "fast"}
            )
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'opencv' or 'torchvision'")
        
        # Verify the output video
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, output_path
        else:
            print(f"Error: Output video file {output_path} is empty or doesn't exist")
            return False, None
            
    except Exception as e:
        print(f"Error saving lip frames to video {output_path}: {str(e)}")
        return False, None

def extract_and_save_lip_video(video_path, output_path, 
                              face_predictor_path=None, 
                              cnn_detector_path=None, 
                              mean_face_path=None, 
                              width_roi=96, 
                              height_roi=96, 
                              fps=25,
                              method='opencv'):
    """
    Extract lip regions from a video file and save them as a new video.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video (.mp4)
        face_predictor_path: Path to the face predictor model file
        cnn_detector_path: Path to the CNN face detector model file
        mean_face_path: Path to the mean face reference file
        width_roi: Width of the lip crop ROI
        height_roi: Height of the lip crop ROI
        fps: Frame rate of the output video, defaults to 25fps
        method: Method to use for creating video ('opencv' or 'torch')
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Extract lip frames
        lip_frames = extract_lip_frames(
            video_path, 
            face_predictor_path, 
            cnn_detector_path, 
            mean_face_path,
            width_roi=width_roi,
            height_roi=height_roi
        )
        
        # Save to video
        return save_lip_frames_to_video(lip_frames, output_path, fps=fps, method=method)
        
    except Exception as e:
        print(f"Error extracting and saving lip video from {video_path}: {str(e)}")
        return False, None

if __name__ == "__main__":
    video_path = "../example/EN2001a-A-21.39-25.86-video.mp4"
    output_path = "../example/EN2001a-A-21.39-25.86-lip_video.mp4"
    face_predictor_path = FACE_PREDICTOR_PATH
    cnn_detector_path = CNN_DETECTOR_PATH
    mean_face_path = MEAN_FACE_PATH
    # process_video_for_av_hubert(video_path, face_predictor_path, cnn_detector_path, mean_face_path)
    
    # Extract and save lip video
    extract_and_save_lip_video(video_path, output_path, 
                             face_predictor_path, cnn_detector_path, mean_face_path,
                             method='torch')