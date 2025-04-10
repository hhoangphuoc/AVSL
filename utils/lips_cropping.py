# THIS SOURCE CODE IS BASED FROM AV-HuBERT implementation at:
# https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/align_mouth.py
# and 
# https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/detect_landmark.py


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import dlib
from skimage import transform as skitf

def detect_landmarks(image, detector, cnn_detector, predictor):
    """
    Detect facial landmarks in an image.
    
    Args:
        image: Input image
        detector: dlib face detector
        cnn_detector: dlib CNN face detector (backup)
        predictor: dlib face landmark predictor
        
    Returns:
        numpy.ndarray: Array of 68 facial landmarks coordinates
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    rects = detector(gray, 1)
    if len(rects) == 0:
        rects = cnn_detector(gray)
        rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def linear_interpolate(landmarks, start_idx, stop_idx):
    """
    Linearly interpolate landmarks between two valid detections.
    
    Args:
        landmarks: List of landmark arrays (may contain None for frames where detection failed)
        start_idx: Index of the first valid detection
        stop_idx: Index of the last valid detection
        
    Returns:
        landmarks: Updated list with interpolated values
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """
    Interpolate missing landmarks.
    
    Args:
        landmarks: List of landmark arrays (may contain None for frames where detection failed)
        
    Returns:
        landmarks: Updated list with interpolated values
    """
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    if not valid_frames_idx:
        return None
    
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    # Handle frames at the beginning or end that failed to be detected
    if valid_frames_idx:
        if valid_frames_idx[0] > 0:
            for idx in range(0, valid_frames_idx[0]):
                landmarks[idx] = landmarks[valid_frames_idx[0]]
        if valid_frames_idx[-1] < len(landmarks) - 1:
            for idx in range(valid_frames_idx[-1] + 1, len(landmarks)):
                landmarks[idx] = landmarks[valid_frames_idx[-1]]
    return landmarks

def warp_img(src, dst, img, std_size):
    """
    Apply affine transformation to align the face.
    
    Args:
        src: Source landmarks
        dst: Destination landmarks (reference)
        img: Input image
        std_size: Output size
        
    Returns:
        tuple: (warped image, transformation)
    """
    tform = skitf.estimate_transform('similarity', src, dst)
    warped = skitf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    """
    Apply a pre-computed transformation to an image.
    
    Args:
        transform: Transformation to apply
        img: Input image
        std_size: Output size
        
    Returns:
        numpy.ndarray: Transformed image
    """
    warped = skitf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped

def cut_patch(img, landmarks, height, width, threshold=5):
    """
    Cut a patch from the image centered on the mean of landmarks.
    
    Args:
        img: Input image
        landmarks: Array of landmarks to center on
        height: Half-height of the patch
        width: Half-width of the patch
        threshold: Error threshold
        
    Returns:
        numpy.ndarray: Cropped patch
    """
    center_x, center_y = np.mean(landmarks, axis=0)
    
    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')
    
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')
    
    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                        int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def create_dlib_detectors(face_predictor_path, cnn_detector_path):
    """
    Create dlib face detector and predictor objects.
    If CUDA is available, the CNN detector will use GPU acceleration.
    
    Args:
        face_predictor_path: Path to the face predictor model file
        cnn_detector_path: Path to the CNN face detector model file
        
    Returns:
        tuple: (detector, cnn_detector, predictor)
    """
    # Check if CUDA is available for dlib
    cuda_available = dlib.DLIB_USE_CUDA
    print(f"DLIB CUDA available: {cuda_available}")
    
    # Standard HOG-based detector (CPU only)
    detector = dlib.get_frontal_face_detector()
    
    # CNN detector (can use GPU if available)
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    
    # Set number of CUDA devices for CNN detector if available
    if cuda_available:
        try:
            # Get number of CUDA devices
            num_cuda_devices = dlib.cuda.get_num_devices()
            print(f"Number of CUDA devices available for dlib: {num_cuda_devices}")
            
            if num_cuda_devices > 0:
                # Set the device to use
                dlib.cuda.set_device(0)
                print("Enabled GPU acceleration for dlib CNN face detector")
        except Exception as e:
            print(f"Error setting up CUDA for dlib: {e}")
    
    # Landmark predictor (CPU only)
    predictor = dlib.shape_predictor(face_predictor_path)
    
    return detector, cnn_detector, predictor