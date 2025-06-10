"""
This script contains function to process video source input, including: 
    - segmenting video based on transcript_segments timestamps, video is resampled to 25fps, mp4 format
    - extracting lip-reading region of video with 96x96 resolution
    - using GPU acceleration for face detection and OpenCV operations where possible
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ffmpeg
import cv2
import numpy as np
from collections import deque
from tqdm import tqdm
import gc  # For garbage collection

from utils import (
    create_dlib_detectors, 
    detect_landmarks, 
    landmarks_interpolate, 
    warp_img, 
    apply_transform, 
    cut_patch
)
from preprocess.constants import CNN_DETECTOR_PATH, FACE_PREDICTOR_PATH, MEAN_FACE_PATH

# Check if CUDA-enabled OpenCV is available
OPENCV_CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"OpenCV CUDA available: {OPENCV_CUDA_AVAILABLE}")

# ================================================================================================================
#                           FUNCTIONS FOR LOADING VIDEO FRAMES  
# ================================================================================================================

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
                (ffmpeg
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

def load_video(video_path, to_grayscale=True, max_frames=None, adaptive_memory=False):
    """
    Load a video file from `video_path` and extract all frames as grayscale. 
    This is used for capturing sequence of frames from video and used for lip-reading.
    
    Args:
        video_path: Path to the video file
        to_grayscale: Whether to convert frames to grayscale
        max_frames: Maximum number of frames to load. If None, load all.
        adaptive_memory: Whether to use adaptive memory management (overrides max_frames)
        
    Returns:
        numpy.ndarray: Array of frames with shape [T, H, W] if grayscale or [T, H, W, 3] if RGB
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties for adaptive memory calculation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame stride based on memory considerations
        frame_stride = 1
        actual_max_frames = total_frames
        
        if adaptive_memory:
            # Estimate memory per frame (in MB)
            bytes_per_pixel = 1 if to_grayscale else 3
            frame_memory_mb = (width * height * bytes_per_pixel) / (1024 * 1024)
            
            # Get available system memory and limit to a reasonable portion (25%)
            try:
                import psutil
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.25
                print(f"Available memory: {available_memory_mb:.1f} MB, Frame size: {frame_memory_mb:.2f} MB/frame")
            except ImportError:
                # If psutil not available, use conservative default
                available_memory_mb = 1024  # Assume 1GB available
                print(f"psutil not available, assuming {available_memory_mb:.1f} MB available memory")
                
            # Calculate safe number of frames based on memory
            safe_frames = int(available_memory_mb / (frame_memory_mb * 2))  # *2 as safety factor
            actual_max_frames = min(total_frames, safe_frames)
            
            print(f"Video has {total_frames} frames, memory-safe limit: {actual_max_frames} frames")
            
            if actual_max_frames < total_frames:
                # If we need to limit frames, compute a sensible stride
                if actual_max_frames < 30:
                    # Ensure we have at least 30 frames for lip tracking
                    actual_max_frames = min(30, total_frames)
                
                frame_stride = max(1, total_frames // actual_max_frames)
        elif max_frames is not None and total_frames > max_frames:
            # Use provided max_frames
            actual_max_frames = max_frames
            print(f"Video has {total_frames} frames, limiting to {max_frames} frames.")
            frame_stride = max(1, total_frames // max_frames)
        
        print(f"Loading {actual_max_frames} frames with stride {frame_stride}")
            
        frames = []
        current_frame_index = 0
        
        with tqdm(total=actual_max_frames, desc="Loading frames", unit="frame") as pbar:
            while True:
                if len(frames) >= actual_max_frames:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    break # End of video or error
                
                if to_grayscale:
                    # Use CUDA for color conversion if available
                    if OPENCV_CUDA_AVAILABLE:
                        try:
                            # Convert to grayscale using CUDA
                            gpu_frame = cv2.cuda_GpuMat()
                            gpu_frame.upload(frame)
                            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                            frame = gpu_gray.download()
                        except Exception as e:
                            # Fallback to CPU if CUDA fails
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    # Convert from BGR (OpenCV default) to RGB
                    if OPENCV_CUDA_AVAILABLE:
                        try:
                            # Convert to RGB using CUDA
                            gpu_frame = cv2.cuda_GpuMat()
                            gpu_frame.upload(frame)
                            gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                            frame = gpu_rgb.download()
                        except Exception as e:
                            # Fallback to CPU if CUDA fails
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                frames.append(frame)
                pbar.update(1)
                
                current_frame_index += frame_stride
                if current_frame_index >= total_frames:
                     break # Prevent infinite loops on bad videos

        cap.release()
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            # Return an empty array with expected dimensions if possible
            # This needs width/height, maybe return None or empty list?
            return np.array([]) # Return empty numpy array
            
        return np.stack(frames)
        
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        # Return empty array on error
        return np.array([])

#=================================================================================================================

def process_video_for_av_hubert(video_path, 
                                face_predictor_path=None, 
                                cnn_detector_path=None, 
                                mean_face_path=None,
                                normalize=True, 
                                image_mean=0.0, 
                                image_std=1.0, 
                                image_crop_size=88,
                                batch_size=16):
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
        batch_size: Size of frame batches for processing
        
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
            batch_size=batch_size
        )
        
        return lip_frames
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        raise e


# ================================================================================================================
#                           EXTRACTING AND SAVING LIP FRAMES  
# ================================================================================================================

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
                       max_frames=300,  # Limit frames for memory
                       adaptive_memory=False):  # Added adaptive memory option
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
        batch_size: Size of frame batches for processing (within single video)
        max_frames: Maximum number of frames to process (limits memory usage)
        adaptive_memory: Whether to adaptively determine max frames based on available memory
        
    Returns:
        numpy.ndarray: Array of lip frames with shape [T, H, W] if grayscale or [T, H, W, 3] if RGB, or empty array on failure.
    """
    try:
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
            return np.array([]) # Return empty array
        
        # Create detectors with CUDA if available
        try:
            detector, cnn_detector, predictor = create_dlib_detectors(face_predictor_path, cnn_detector_path)
        except Exception as e:
            print(f"Error creating face detectors: {str(e)}")
            return np.array([]) # Return empty array
        
        # Load video frames using OpenCV, using adaptive memory management if requested
        frames = load_video(video_path, to_grayscale=to_grayscale, 
                           max_frames=max_frames, adaptive_memory=adaptive_memory)
        
        if frames.size == 0:
             print(f"No frames loaded from {video_path}, cannot extract lips.")
             return np.array([]) # Return empty array
             
        # Save the frames length for later use
        frames_length = len(frames)
        window_margin = min(frames_length, 12)  # Limit window margin for long videos
        
        # Process frames sequentially to detect landmarks
        landmarks = []
        for i, frame in enumerate(tqdm(frames, desc="Detecting landmarks", unit="frame")):
            try:
                if not to_grayscale:
                    # Make a temporary grayscale copy for face detection
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    landmark = detect_landmarks(gray_frame, detector, cnn_detector, predictor)
                else:
                    landmark = detect_landmarks(frame, detector, cnn_detector, predictor)
                landmarks.append(landmark)
            except Exception as e:
                print(f"Error detecting landmark in frame {i}: {e}")
                landmarks.append(None)
        
        # Clean up memory
        del frames
        gc.collect()
        
        # Interpolate landmarks where detection failed
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            print(f"No face detected or landmarks couldn't be interpolated in video {video_path}")
            return np.array([]) # Return empty array
        
        # Define stable points for alignment
        stablePntsIDs = [33, 36, 39, 42, 45]
        STD_SIZE = (300, 300)
        
        # Reload video frames (we deleted them to save memory)
        frames = load_video(video_path, to_grayscale=to_grayscale, 
                           max_frames=max_frames, adaptive_memory=adaptive_memory)
        
        if frames.size == 0:
             print(f"Failed to reload frames from {video_path}")
             return np.array([]) # Return empty array
        
        # Process frames to extract lip regions
        frame_idx = 0
        sequence = []
        margin = window_margin
        q_frame, q_landmarks = deque(), deque()
        valid_transform_found = False
        trans = None # Initialize trans
        
        for frame_idx, frame in enumerate(frames):
            if landmarks[frame_idx] is None: # Skip if landmarks are None
                 continue
                 
            q_landmarks.append(landmarks[frame_idx])
            q_frame.append(frame)
            
            if len(q_frame) == margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()
                
                # Apply affine transformation
                try:
                    trans_frame, trans_matrix = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                                mean_face_landmarks[stablePntsIDs, :],
                                                cur_frame,
                                                STD_SIZE)
                    trans = trans_matrix # Store the valid transformation matrix
                    valid_transform_found = True
                    trans_landmarks = trans(cur_landmarks)
                except Exception as warp_error:
                     print(f"Error during warp_img for frame {frame_idx}: {warp_error}")
                     continue # Skip this frame if warping fails
                
                # Crop mouth patch
                try:
                    cropped_patch = cut_patch(trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        height_roi//2,
                                        width_roi//2)
                    sequence.append(cropped_patch)
                except Exception as e:
                    print(f"Error cropping frame {frame_idx}: {str(e)}")
                    # If cropping fails, skip adding this frame
                    pass
        
        # Process remaining frames in the deque
        while q_frame:
            cur_frame = q_frame.popleft()
            cur_landmarks = q_landmarks.popleft()
            
            if cur_landmarks is None or not valid_transform_found:
                 continue # Skip if landmarks are None or no valid transform was found
                 
            # Transform frame using the last valid transformation
            trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            
            # Crop mouth patch
            try:
                cropped_patch = cut_patch(trans_frame,
                                    trans_landmarks[start_idx:stop_idx],
                                    height_roi//2,
                                    width_roi//2)
                sequence.append(cropped_patch)
            except Exception as e:
                print(f"Error cropping remaining frame: {str(e)}")
                pass
        
        # Clean up memory
        del frames, landmarks, q_frame, q_landmarks
        gc.collect()
        
        if not sequence:
            print(f"No lip sequences generated for {video_path}")
            return np.array([])
            
        return np.array(sequence)
    
    except Exception as e:
        print(f"Critical Error extracting lip frames from {video_path}: {str(e)}")
        # Return an empty array that matches the expected format
        return np.array([])

def save_lip_frames_to_video(lip_frames, output_path, fps=25):
    """
    Save lip-cropped frames as an MP4 video using CPU-based OpenCV. 
    This function exporting the lip-cropped frames as a video file.
    
    Args:
        lip_frames: numpy.ndarray of shape [T, H, W] for grayscale or [T, H, W, 3] for RGB
        output_path: Path to save the output video (.mp4)
        fps: Frame rate of the output video, defaults to 25fps
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    if lip_frames.size == 0 or len(lip_frames.shape) < 3:
        print(f"Cannot save video {output_path}: Input lip_frames are empty or invalid shape {lip_frames.shape}.")
        return False, None
        
    try:
        # Check that output file ends with .mp4
        if not output_path.lower().endswith('.mp4'):
            original_output = output_path
            output_path = f"{os.path.splitext(original_output)[0]}.mp4"
        
        # Check if frames are grayscale or RGB
        is_grayscale = len(lip_frames.shape) == 3  # [T, H, W]
        
        # Get dimensions
        num_frames, height, width = lip_frames.shape[:3]
        if height <= 0 or width <= 0:
            print(f"Invalid frame dimensions for {output_path}: {width}x{height}")
            return False, None
            
        print(f"Writing {num_frames} frames to {output_path} using OpenCV (dimensions: {width}x{height})")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
             print(f"Error: Could not open video writer for path: {output_path}")
             return False, None
             
        # Process and write each frame
        for frame in lip_frames:
            try:
                if is_grayscale:
                    # Convert grayscale to BGR (OpenCV expects BGR for writing)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    # Convert RGB to BGR (OpenCV expects BGR for writing)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            except cv2.error as frame_error:
                 print(f"OpenCV error writing frame to {output_path}: {frame_error}")
                 # Continue trying to write other frames if possible
                 pass
        
        # Release resources
        out.release()
        
        # Verify the output video
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, output_path
        else:
            # Check if file exists but is empty
            if os.path.exists(output_path):
                print(f"Error: Output video file {output_path} was created but is empty.")
                os.remove(output_path) # Remove empty file
            else:
                print(f"Error: Output video file {output_path} was not created.")
            return False, None
            
    except Exception as e:
        print(f"Error saving lip frames to video {output_path}: {str(e)}")
        # Clean up partially created file if it exists
        if 'out' in locals() and out.isOpened():
             out.release()
        if os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except OSError:
                 pass # Ignore error if file can't be removed
        return False, None


# ================================================================================================================
#                           EXTRACTING AND SAVING LIP VIDEO
# ================================================================================================================

def extract_and_save_lip_video(video_path, output_path, 
                              face_predictor_path=None, 
                              cnn_detector_path=None, 
                              mean_face_path=None, 
                              width_roi=96, 
                              height_roi=96, 
                              fps=25,
                              to_grayscale=True, # Default to grayscale for efficiency
                              batch_size=16,
                              max_frames=300, # Add max_frames here too
                              adaptive_memory=False): # Add adaptive memory option
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
        batch_size: Size of frame batches for processing within each video
        max_frames: Maximum number of frames to process from the video
        adaptive_memory: Whether to adaptively determine max frames based on memory
        
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    # For debugging, print the filename
    print(f"Processing video: {os.path.basename(video_path)}")
    try:
        # Check if file exists before processing
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False, None
            
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
            max_frames=max_frames,
            adaptive_memory=adaptive_memory
        )
        
        # Check if extraction was successful
        if lip_frames.size == 0:
            print(f"Lip frame extraction failed for {video_path}")
            return False, None
            
        # Save to video
        success, output_file = save_lip_frames_to_video(lip_frames, output_path, fps=fps)
        
        # Free up memory explicitly
        del lip_frames
        gc.collect()
        
        return success, output_file
        
    except Exception as e:
        print(f"Error in extract_and_save_lip_video for {video_path}: {str(e)}")
        return False, None

# ================================================================================================================
#                           IMPROVED SEQUENTIAL/MULTIPROCESSING PROCESSING
# ================================================================================================================

import multiprocessing as mp
from multiprocessing.pool import Pool
import signal
import random

# Global variable to store process pool
_process_pool = None

def init_worker():
    """
    Initialize worker process - ignore SIGINT so the parent can handle it
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_video_task(task_data):
    """
    Task function for processing a single video in a worker process.
    This function is used for both `sequential` and `multiprocessing` processing, 
    depending on the `use_multiprocessing` flag. 
    This function outputs a tuple of `(segment_id, success_flag, output_file_path)`, 
    corresponding to the lip extracted video from the video segment, 
    whether the lip extraction was successful, and the path to the output video file.

    If `use_multiprocessing` is `True`, the function will be called by the multiprocessing pool. 
    Then, multiple videos segments will be processed in parallel.

    If `use_multiprocessing` is `False`, otherwise, each video segment will be processed sequentially.
    
    Args:
        task_data: Tuple containing (video_path, output_path, segment_id, kwargs)
        
    Returns:
        Tuple of (segment_id, success_flag, output_file_path)
    """
    try:
        video_path, output_path, segment_id, kwargs = task_data
        # Print task info
        print(f"Worker processing: {os.path.basename(video_path)}")
        
        # Process video
        success, output_file = extract_and_save_lip_video(video_path, output_path, **kwargs)
        
        # Force garbage collection
        gc.collect()
        
        return segment_id, success, output_file
    except Exception as e:
        print(f"Error in worker process for {task_data[0]}: {str(e)}")
        return task_data[2], False, None

def batch_process_lip_videos(video_paths, output_paths, 
                            num_workers=None, 
                            max_tasks_per_child=10,
                            use_multiprocessing=True,
                            adaptive_memory=False,
                            **kwargs):
    """
    NOTE: THIS IS THE MAIN FUNCTION FOR LIP VIDEO EXTRACTION, WHICH CALLED BY `process_in_chunks.py`
    
    Process multiple videos sequentially/parallelly to extract lip videos.
    This function uses GPU acceleration for face detection and OpenCV operations
    but processes each video one at a time.

    Args:
        video_paths: List of video paths to process or Dictionary {segment_id: (success, video_path)}
        output_paths: List of output paths to save results or Directory for outputs if video_paths is a dictionary
        num_workers: Number of worker processes to use (default: CPU count - 1)
        max_tasks_per_child: Maximum number of tasks per worker before respawning
        use_multiprocessing: Whether to use multiprocessing or sequential processing
        adaptive_memory: Whether to use adaptive memory management (avoids fixed max_frames limits)
        **kwargs: Additional arguments to pass to extract_and_save_lip_video
        
    Returns:
        Tuple of (lip_results_dict, successful_count)
         - lip_results_dict: Dict {segment_id: (success, lip_video_path)}
         - successful_count: Number of successfully processed videos
    """
    # Convert input format if it's a dictionary
    segment_ids = []
    if isinstance(video_paths, dict):
        # Extract lists from dictionary
        video_segment_results = video_paths
        video_paths = []
        segment_ids = []
        lip_video_dir = output_paths
        output_paths = []
        
        # Prepare lists for processing
        for segment_id, (success, video_path) in video_segment_results.items():
            if success and video_path and os.path.exists(video_path):
                lip_output_path = os.path.join(lip_video_dir, f"{segment_id}-lip.mp4")
                video_paths.append(video_path)
                output_paths.append(lip_output_path)
                segment_ids.append(segment_id)
            else:
                print(f"Skipping segment {segment_id}: No valid video path provided or file missing.")
        
        if not video_paths:
            print("No valid video paths found to process.")
            return {}, 0
    
    # If adaptive_memory is enabled, set in kwargs
    if adaptive_memory:
        kwargs['adaptive_memory'] = True
        if 'max_frames' in kwargs:
            print(f"Adaptive memory enabled, ignoring max_frames={kwargs['max_frames']}")
    
    # Prepare tasks for processing
    total_videos = len(video_paths)
    tasks = []
    for i, (video_path, output_path) in enumerate(zip(video_paths, output_paths)):
        # For dictionary inputs, use segment_id; otherwise, use index
        segment_id = segment_ids[i] if segment_ids else f"video_{i}"
        tasks.append((video_path, output_path, segment_id, kwargs))
    
    # Determine processing approach
    if use_multiprocessing and total_videos > 1:
        # Use multiprocessing for multiple videos
        if num_workers is None:
            # Leave one CPU for the main process
            num_workers = max(1, mp.cpu_count() - 1)
            
        print(f"Processing {total_videos} videos using {num_workers} worker processes")
        print(f"Worker processes will be recycled after {max_tasks_per_child} tasks")
        print(f"Adaptive memory: {adaptive_memory}")
        
        # Initialize results
        results = []
        
        try:
            # Create process pool with worker initialization
            with Pool(processes=num_workers, 
                     initializer=init_worker, 
                     maxtasksperchild=max_tasks_per_child) as pool:
                
                # Process tasks with progress bar
                for result in tqdm(pool.imap_unordered(process_video_task, tasks), 
                                  total=len(tasks), 
                                  desc="Processing videos"):
                    # Store result
                    results.append(result)
                    
                    # Print progress regularly
                    successful = sum(1 for _, success, _ in results if success)
                    if len(results) % 10 == 0 or len(results) == total_videos:
                        print(f"Processed {len(results)}/{total_videos} videos. Success rate: {successful/len(results)*100:.1f}%")
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving partial results...")
        except Exception as e:
            print(f"Error in processing pool: {str(e)}")
            print("Saving partial results...")
            
    else:
        # Use sequential processing
        print(f"Sequential processing {total_videos} videos")
        print(f"Adaptive memory: {adaptive_memory}")
        results = []
        
        try:
            # Process tasks one by one
            for task in tqdm(tasks, desc="Processing videos"):
                result = process_video_task(task)
                results.append(result)
                
                # Print progress regularly
                successful = sum(1 for _, success, _ in results if success)
                if len(results) % 10 == 0 or len(results) == total_videos:
                    print(f"Processed {len(results)}/{total_videos} videos. Success rate: {successful/len(results)*100:.1f}%")
                    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving partial results...")
    
    # Process results
    successful_count = 0
    lip_results_dict = {}
    
    for segment_id, success, output_file in results:
        lip_results_dict[segment_id] = (success, output_file)
        if success:
            successful_count += 1
    
    # If original input was a dictionary, ensure all segments are represented
    if isinstance(video_paths, dict):
        for segment_id in video_segment_results.keys():
            if segment_id not in lip_results_dict:
                lip_results_dict[segment_id] = (False, None)
    
    return lip_results_dict, successful_count

# ================================================================================================================  

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Processing and Lip Extraction')
    parser.add_argument('--video_path', type=str, help='Path to input video')
    parser.add_argument('--output_path', type=str, help='Path to output video')
    parser.add_argument('--to_grayscale', action='store_true', default=True, help='Process in grayscale')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for frame processing within each video')
    parser.add_argument('--max_frames', type=int, default=300, help='Max frames to load per video')
    
    args = parser.parse_args()
    
    video_path = args.video_path or "../example/EN2001a-A-21.39-25.86-video.mp4"
    output_path = args.output_path or "../example/EN2001a-A-21.39-25.86-lip_video_gpu.mp4"
    face_predictor_path = FACE_PREDICTOR_PATH
    cnn_detector_path = CNN_DETECTOR_PATH
    mean_face_path = MEAN_FACE_PATH
    
    print(f"Processing video: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Grayscale: {args.to_grayscale}")
    print(f"CUDA available for OpenCV: {OPENCV_CUDA_AVAILABLE}")
    print(f"Max frames: {args.max_frames}")
    
    # Extract and save lip video 
    success, result_path = extract_and_save_lip_video(
        video_path, 
        output_path, 
        face_predictor_path, 
        cnn_detector_path, 
        mean_face_path,
        to_grayscale=args.to_grayscale,
        batch_size=args.batch_size,
        max_frames=args.max_frames
    )
    
    if success:
        print(f"Successfully processed lip video saved to {result_path}")
    else:
        print(f"Failed to process lip video for {video_path}")