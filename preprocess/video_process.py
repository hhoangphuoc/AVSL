"""
This script contains function to process video source input, including: 
    - segmenting video based on transcript_segments timestamps, video is resampled to 25fps, mp4 format
    - extracting lip-reading region of video with 96x96 resolution
    - batch processing for faster extraction
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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

from utils import (
    create_dlib_detectors, 
    detect_landmarks, 
    landmarks_interpolate, 
    warp_img, 
    apply_transform, 
    cut_patch
)
from preprocess.constants import CNN_DETECTOR_PATH, FACE_PREDICTOR_PATH, MEAN_FACE_PATH

# Global variables for GPU usage
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Number of worker processes/threads for parallel processing
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1) #Leave 1 core for other processes

# ================================================================================================================
#                           FUNCTIONS FOR LOADING VIDEO FRAMES  
# ================================================================================================================

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
    # Call batch segmentation with a single segment
    segments = [(start_time, end_time, video_output_file)]
    results = batch_segment_video(video_file, segments)
    
    # Return the result for the single segment
    if results and len(results) > 0:
        return results[0]
    return False, None

def batch_segment_video(video_file, segments):
    """
    Process multiple video segments from the same source file more efficiently.
    Probes the video file once and extracts all segments.
    
    Args:
        video_file: Path to the input video file
        segments: List of tuples (start_time, end_time, output_file)
    
    Returns:
        List of tuples (success_flag, output_file_path) for each segment
    """
    if not segments:
        return []
        
    results = []
    
    try:
        # Probe video only once
        print(f"Probing video file: {video_file}")
        probe = ffmpeg.probe(video_file)
        video_duration = float(probe['format']['duration'])
        
        # Process each segment
        for start_time, end_time, video_output_file in segments:
            try:
                # Ensure output is mp4
                if not video_output_file.lower().endswith('.mp4'):
                    original_output = video_output_file
                    video_output_file = f"{os.path.splitext(video_output_file)[0]}.mp4"
                    print(f"Changing output format from {original_output} to {video_output_file}")
                    
                # Validate timestamps
                if start_time < 0:
                    print(f"Warning: Negative start time {start_time} for {video_file}, setting to 0")
                    start_time = 0
                
                if end_time > video_duration:
                    print(f"Warning: End time {end_time} exceeds video duration {video_duration} for {video_file}, truncating")
                    end_time = video_duration
                    
                if start_time >= end_time:
                    print(f"Warning: Invalid video segment {start_time}-{end_time} for {video_file}")
                    results.append((False, None))
                    continue
                    
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
                    results.append((True, video_output_file))
                else:
                    print(f"Error: Output video file {video_output_file} is empty or doesn't exist")
                    results.append((False, None))
                    
            except ffmpeg.Error as e:
                print(f"Error segmenting video {video_file} at {start_time}-{end_time}: {e.stderr}")
                results.append((False, None))
            except Exception as e:
                print(f"Unexpected error processing video segment {start_time}-{end_time} from {video_file}: {str(e)}")
                results.append((False, None))
        
    except Exception as e:
        print(f"Error probing video {video_file}: {str(e)}")
        # If we can't probe the video file, all segments fail
        return [(False, None) for _ in segments]
    
    return results

def load_video(video_path, to_grayscale=True):
    """
    Load a video file from `video_path` and extract all frames as grayscale. 
    This is used for capturing sequence of frames from video and used for lip-reading.
    
    Args:
        video_path: Path to the video file
        to_grayscale: Whether to convert frames to grayscale
        
    Returns:
        numpy.ndarray: Array of frames with shape [T, H, W] if grayscale or [T, H, W, 3] if RGB
    """
    try:
        # Use more efficient FFMPEG-based loading if possible
        try:
            if USE_GPU and torch.cuda.is_available():
                # Using torchvision's GPU accelerated video reader
                import torchvision
                video_tensor = torchvision.io.read_video(video_path, pts_unit='sec')[0]  # Returns [T, H, W, C]
                video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
                
                if to_grayscale:
                    # Convert to grayscale using GPU
                    grayscale_transform = transforms.Grayscale(num_output_channels=1)
                    video_tensor = grayscale_transform(video_tensor).squeeze(1)  # [T, H, W]
                    frames = video_tensor.cpu().numpy()
                else:
                    # Convert to RGB (already in RGB)
                    frames = video_tensor.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
                
                return frames
        except (ImportError, RuntimeError) as e:
            # Fall back to OpenCV if torchvision IO fails
            print(f"Torchvision video loading failed, falling back to OpenCV: {str(e)}")
        
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                if to_grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    # Convert from BGR (OpenCV default) to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

#=================================================================================================================

def video_to_tensor(frames, normalize=True, image_mean=0.0, image_std=1.0, image_crop_size=88):
    """
    Convert video frames to tensor format suitable for AV-HuBERT. 
    The frames are grayscale, and being normalised to [0, 1], 
    cropped to `image_crop_size=(88 x 88)`, and then expanded to `dims = [T, H, W, 1]`.
    
    Args:
        frames: numpy.ndarray of shape [T, H, W]
        normalize: Whether to normalize pixel values
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        image_crop_size: Size to crop the frames to
        
    Returns:
        numpy.ndarray: Processed frames with shape [T, H, W, 1]
    """
    # Use GPU if available for transformation
    if USE_GPU and isinstance(frames, np.ndarray):
        # Convert to torch tensor for faster processing on GPU
        frames_tensor = torch.from_numpy(frames).to('cuda')
        
        # Normalize to [0, 1]
        frames_tensor = frames_tensor.float() / 255.0
        
        # Center crop using torch operations
        if frames_tensor.shape[1] != image_crop_size or frames_tensor.shape[2] != image_crop_size:
            center_crop = transforms.CenterCrop((image_crop_size, image_crop_size))
            if len(frames_tensor.shape) == 3:  # [T, H, W]
                frames_tensor = frames_tensor.unsqueeze(1)  # Add channel dim [T, 1, H, W]
                frames_tensor = center_crop(frames_tensor)
                frames_tensor = frames_tensor.squeeze(1)  # Remove channel dim [T, H, W]
            else:  # [T, H, W, C]
                frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
                frames_tensor = center_crop(frames_tensor)
                frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # [T, H, W, C]
        
        # Normalize with mean/std if requested
        if normalize:
            frames_tensor = (frames_tensor - image_mean) / image_std
        
        # Add channel dimension if needed
        if len(frames_tensor.shape) == 3:  # [T, H, W]
            frames_tensor = frames_tensor.unsqueeze(-1)  # [T, H, W, 1]
        
        # Return as numpy array
        return frames_tensor.cpu().numpy()
    else:
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
                                image_crop_size=88,
                                batch_size=16,
                                use_parallel=True):
    """
    Process a HuggingFace Video object (a video file) for AV-HuBERT Video Encoder.
    The video frames will be converted to grayscale for AV-HuBERT to process
    
    Args:
        video_path: Path to the video file
        face_predictor_path: Path to the face predictor model file
        cnn_detector_path: Path to the CNN face detector model file
        mean_face_path: Path to the mean face reference file
        normalize: Whether to normalize pixel values
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        image_crop_size: Size to crop the frames to
        batch_size: Size of frame batches for parallel processing
        use_parallel: Whether to use parallel processing
        
    Returns:
        numpy.ndarray: Processed video features with shape [T, H, W, 1]
    """
    
    try:
        
        # Extract lip frames
        lip_frames = extract_lip_frames(
            video_path, 
            face_predictor_path, 
            cnn_detector_path, 
            mean_face_path, 
            to_grayscale=True,
            batch_size=batch_size,
            use_parallel=use_parallel
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


# ================================================================================================================
#                           EXTRACTING AND SAVING LIP FRAMES  
# ================================================================================================================

def process_frame_batch(frames_batch, detector, cnn_detector, predictor, to_grayscale):
    """
    Process a batch of frames to detect landmarks in parallel
    
    Args:
        frames_batch: List of frames to process
        detector: dlib face detector
        cnn_detector: dlib CNN face detector
        predictor: dlib shape predictor
        to_grayscale: Whether frames are in grayscale
        
    Returns:
        List of detected landmarks for each frame
    """
    landmarks_batch = []
    for frame in frames_batch:
        # If frames are RGB, convert to grayscale temporarily for landmark detection
        if not to_grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            landmark = detect_landmarks(gray_frame, detector, cnn_detector, predictor)
        else:
            landmark = detect_landmarks(frame, detector, cnn_detector, predictor)
        landmarks_batch.append(landmark)
    return landmarks_batch

def extract_lip_frames(video_path, 
                       face_predictor_path=None, 
                       cnn_detector_path=None, 
                       mean_face_path=None, 
                       width_roi=96, 
                       height_roi=96, 
                       start_idx=48, 
                       stop_idx=68,
                       to_grayscale=True,
                       batch_size=16,
                       use_parallel=True):
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
        to_grayscale: Whether to process and return grayscale frames
        batch_size: Size of frame batches for parallel processing
        use_parallel: Whether to use parallel processing
        
    Returns:
        numpy.ndarray: Array of lip frames with shape [T, H, W] if grayscale or [T, H, W, 3] if RGB
    """
    # Set default paths if not provided
    if face_predictor_path is None:
        face_predictor_path = FACE_PREDICTOR_PATH
    if cnn_detector_path is None:
        cnn_detector_path = CNN_DETECTOR_PATH
    if mean_face_path is None:
        mean_face_path = MEAN_FACE_PATH
        
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
    
    # Load video frames (either grayscale or RGB based on to_grayscale)
    frames = load_video(video_path, to_grayscale=to_grayscale)
    
    # Detect landmarks for each frame - using parallel processing if enabled
    if use_parallel and len(frames) > batch_size:
        landmarks = []
        # Split frames into batches
        frame_batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Process batches in parallel
            futures = []
            for batch in frame_batches:
                future = executor.submit(
                    process_frame_batch, 
                    batch, 
                    detector, 
                    cnn_detector, 
                    predictor, 
                    to_grayscale
                )
                futures.append(future)
            
            # Collect results
            for future in tqdm(futures, desc="Processing batches", total=len(futures)):
                batch_landmarks = future.result()
                landmarks.extend(batch_landmarks)
    else:
        # Process frames sequentially
        landmarks = []
        for frame in frames:
            # If frames are RGB, convert to grayscale temporarily for landmark detection
            if not to_grayscale:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                landmark = detect_landmarks(gray_frame, detector, cnn_detector, predictor)
            else:
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
                cropped_patch = cut_patch(trans_frame,
                                    trans_landmarks[start_idx:stop_idx],
                                    height_roi//2,
                                    width_roi//2)
                sequence.append(cropped_patch)
            except Exception as e:
                print(f"Error cropping frame {frame_idx}: {str(e)}")
                # If cropping fails, add a black frame to maintain synchronization
                if to_grayscale:
                    sequence.append(np.zeros((height_roi, width_roi), dtype=np.uint8))
                else:
                    sequence.append(np.zeros((height_roi, width_roi, 3), dtype=np.uint8))
        
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
            cropped_patch = cut_patch(trans_frame,
                                trans_landmarks[start_idx:stop_idx],
                                height_roi//2,
                                width_roi//2)
            sequence.append(cropped_patch)
        except Exception as e:
            print(f"Error cropping frame: {str(e)}")
            if to_grayscale:
                sequence.append(np.zeros((height_roi, width_roi), dtype=np.uint8))
            else:
                sequence.append(np.zeros((height_roi, width_roi, 3), dtype=np.uint8))
    
    return np.array(sequence)


def save_lip_frames_to_video(lip_frames, output_path, fps=25):
    """
    Save lip-cropped frames as an MP4 video. 
    This function exporting the lip-cropped frames as a video file.
    
    Args:
        lip_frames: numpy.ndarray of shape [T, H, W] for grayscale or [T, H, W, 3] for RGB
        output_path: Path to save the output video (.mp4)
        fps: Frame rate of the output video, defaults to 25fps
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Check that output file ends with .mp4
        if not output_path.lower().endswith('.mp4'):
            original_output = output_path
            output_path = f"{os.path.splitext(output_path)[0]}.mp4"
        
        # Check if frames are grayscale or RGB
        is_grayscale = len(lip_frames.shape) == 3  # [T, H, W]
        
        # Get dimensions
        if is_grayscale:
            num_frames, height, width = lip_frames.shape
        else:
            num_frames, height, width, channels = lip_frames.shape

        
        # Try to use GPU-accelerated saving if available
        # FIXME: THIS IS NOT WORKING AS EXPECTED, FALLBACK TO OPENCV INSTEAD --------------------------------
        if USE_GPU and torch.cuda.is_available():
            try:
                import torchvision
                
                # Convert frames to tensor for GPU processing
                if is_grayscale:
                    # For grayscale frames, we need to convert to RGB for video writing
                    print("Converting grayscale frames to RGB for GPU video writing")
                    rgb_frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
                    for i in range(num_frames):
                        # Convert each frame from grayscale to RGB
                        rgb_frames[i] = cv2.cvtColor(lip_frames[i], cv2.COLOR_GRAY2RGB)
                    
                    # Convert to torch tensor format for torchvision
                    frames_tensor = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float().to('cuda')
                else:
                    # For RGB frames, convert to torch tensor directly
                    frames_tensor = torch.from_numpy(lip_frames).permute(0, 3, 1, 2).float().to('cuda')
                
                # Write video using torchvision (requires TCHW format)
                print(f"Writing video to {output_path} using GPU acceleration")
                torchvision.io.write_video(
                    output_path,
                    frames_tensor.permute(0, 2, 3, 1).cpu(),  # Convert back to THWC for write_video
                    fps=fps,
                    video_codec="libx264"
                )
                
                # If video was successfully written, return
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Successfully wrote video to {output_path} using GPU")
                    return True, output_path
                
                # Otherwise fall back to OpenCV
                print("GPU video writing failed, falling back to OpenCV")
            except Exception as e:
                print(f"Error using GPU for video writing: {str(e)}")
                print("Falling back to OpenCV")
        
        # Fallback to OpenCV
        print(f"Writing video to {output_path} using OpenCV")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in lip_frames:
            if is_grayscale:
                # Convert grayscale to BGR (OpenCV expects BGR for writing)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                # Convert RGB to BGR (OpenCV expects BGR for writing)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Verify the output video
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully wrote video to {output_path} using OpenCV")
            return True, output_path
        else:
            print(f"Error: Output video file {output_path} is empty or doesn't exist")
            return False, None
            
    except Exception as e:
        print(f"Error saving lip frames to video {output_path}: {str(e)}")
        return False, None


# ================================================================================================================
#                           FUNCTIONS FOR MULTI-PROCESSING LIP VIDEO EXTRACTION
# ================================================================================================================

def extract_and_save_lip_video(video_path, output_path, 
                              face_predictor_path=None, 
                              cnn_detector_path=None, 
                              mean_face_path=None, 
                              width_roi=96, 
                              height_roi=96, 
                              fps=25,
                              to_grayscale=False,
                              batch_size=16,
                              use_parallel=True,
                              use_gpu=None):  # Added use_gpu parameter
    """
    Extract lip regions from a video file and save them as a new video.
    This is the combined function of `extract_lip_frames` and `save_lip_frames_to_video`.

    This function is mainly used for visualisation of lip-reading results.
    The output video is in RGB format by default (`to_grayscale=False`).
    This is not recommended for using for AV-HuBERT, instead, set `to_grayscale=True` if we want to output a grayscale video.
    
    Alternatively, use `process_video_for_av_hubert` for AV-HuBERT video processing.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video (.mp4)
        face_predictor_path: Path to the face predictor model file
        cnn_detector_path: Path to the CNN face detector model file
        mean_face_path: Path to the mean face reference file
        width_roi: Width of the lip crop ROI
        height_roi: Height of the lip crop ROI
        fps: Frame rate of the output video, defaults to 25fps
        to_grayscale: Whether to process frames in grayscale (True) or color (False)
        batch_size: Size of frame batches for parallel processing
        use_parallel: Whether to use parallel processing
        use_gpu: Whether to use GPU acceleration (overrides global setting if provided)
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Set default paths if not provided
        if face_predictor_path is None:
            face_predictor_path = FACE_PREDICTOR_PATH
        if cnn_detector_path is None:
            cnn_detector_path = CNN_DETECTOR_PATH
        if mean_face_path is None:
            mean_face_path = MEAN_FACE_PATH
        
        # Override the global USE_GPU setting if a value is provided
        global USE_GPU
        original_use_gpu = USE_GPU
        if use_gpu is not None:
            USE_GPU = use_gpu and torch.cuda.is_available()
            
        print(f"Extracting lip frames from {video_path} using {'parallel' if use_parallel else 'sequential'} processing")
        print(f"GPU acceleration: {'enabled' if USE_GPU else 'disabled'}")
        
        # Extract lip frames
        lip_frames = extract_lip_frames(
            video_path, 
            face_predictor_path, 
            cnn_detector_path, 
            mean_face_path,
            width_roi=width_roi,
            height_roi=height_roi,
            to_grayscale=to_grayscale,
            batch_size=batch_size,
            use_parallel=use_parallel
        )
        
        # Save to video
        print(f"Saving video to {output_path}")
        result = save_lip_frames_to_video(lip_frames, output_path, fps=fps)
        
        # Restore original USE_GPU setting
        if use_gpu is not None:
            USE_GPU = original_use_gpu
            
        return result
        
    except Exception as e:
        # Restore original USE_GPU setting in case of error
        if use_gpu is not None:
            USE_GPU = original_use_gpu
        print(f"Error extracting and saving lip video from {video_path}: {str(e)}")
        return False, None

def process_videos_in_parallel(video_paths, output_paths, extract_func, **kwargs):
    """
    Process multiple videos in parallel. 
    by executing the `extract_func` for multiple workers in parallel, each handling a `batch_size` of videos.
    This is a inner function for `batch_process_lip_videos`, which created multiple thread processes.
    
    For direct use, using `batch_process_lip_videos` instead.
    
    Args:
        video_paths: List of video paths to process
        output_paths: List of output paths to save results
        extract_func: Function to apply to each video
        **kwargs: Additional arguments to pass to extract_func
        
    Returns:
        List of results from extract_func for each video
    """
    results = []
    
    # If number of videos is small, simply process them with a thread pool instead
    if len(video_paths) < 5:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for video_path, output_path in zip(video_paths, output_paths):
                future = executor.submit(extract_func, video_path, output_path, **kwargs)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing videos", total=len(futures)):
                result = future.result()
                results.append(result)
        return results
    
    # For larger datasets, use process pool for better parallelism
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for video_path, output_path in zip(video_paths, output_paths):
            future = executor.submit(extract_func, video_path, output_path, **kwargs)
            futures.append(future)
        
        for future in tqdm(futures, desc="Processing videos", total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in video processing worker: {str(e)}")
                results.append((False, None))
    
    return results

# MAIN FUNCTION CALL
def batch_process_lip_videos(video_paths, output_paths, use_gpu=False, **kwargs):
    """
    Process multiple videos in parallel to extract lip videos.
    This function is a wrapper of `process_videos_in_parallel`, 
    which created multiple thread processes that handling extraction of multiple videos (`extract_and_save_lip_video`) in parallel.
    
    Args:
        video_paths: List of video paths to process
        output_paths: List of output paths to save lip videos
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments to pass to extract_and_save_lip_video
        
    Returns:
        List of tuples (success_flag, output_file_path) for each video
    """
    # Explicitly add use_gpu to kwargs to ensure it's passed to extract_and_save_lip_video
    kwargs['use_gpu'] = use_gpu
    
    print(f"Batch processing {len(video_paths)} videos with use_gpu={use_gpu}")
    
    # Verification step to ensure all paths are valid
    valid_pairs = []
    valid_indices = []
    
    for i, (video_path, output_path) in enumerate(zip(video_paths, output_paths)):
        if os.path.exists(video_path):
            valid_pairs.append((video_path, output_path))
            valid_indices.append(i)
        else:
            print(f"Warning: Video path does not exist: {video_path}")
    
    # Process only valid pairs
    if valid_pairs:
        valid_video_paths = [p[0] for p in valid_pairs]
        valid_output_paths = [p[1] for p in valid_pairs]
        
        results = process_videos_in_parallel(valid_video_paths, valid_output_paths, extract_and_save_lip_video, **kwargs)
        
        # Create full results list with failed entries for invalid paths
        full_results = [(False, None)] * len(video_paths)
        for idx, result in zip(valid_indices, results):
            full_results[idx] = result
            
        return full_results
    else:
        print("No valid video paths found. Skipping batch processing.")
        return [(False, None)] * len(video_paths)

# ================================================================================================================  

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Processing and Lip Extraction')
    parser.add_argument('--video_path', type=str, help='Path to input video')
    parser.add_argument('--output_path', type=str, help='Path to output video')
    parser.add_argument('--to_grayscale', default=False, help='Process in grayscale')
    parser.add_argument('--use_gpu', default=True, help='Use GPU if available')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for parallel processing')
    parser.add_argument('--use_parallel', action='store_true', default=True, help='Use parallel processing')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes/threads')
    
    args = parser.parse_args()
    
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers
    
    # Override global GPU usage if specified
    if not args.use_gpu:
        USE_GPU = False
    
    video_path = args.video_path or "../example/EN2001a-A-21.39-25.86-video.mp4"
    output_path = args.output_path or "../example/EN2001a-A-21.39-25.86-lip_video.mp4"
    face_predictor_path = FACE_PREDICTOR_PATH
    cnn_detector_path = CNN_DETECTOR_PATH
    mean_face_path = MEAN_FACE_PATH
    
    # Extract and save lip video
    extract_and_save_lip_video(video_path, output_path, 
                             face_predictor_path, 
                             cnn_detector_path, 
                             mean_face_path,
                             to_grayscale=args.to_grayscale,
                             batch_size=args.batch_size,
                             use_parallel=args.use_parallel)