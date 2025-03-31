import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#------------------------------------------------------------------------------
# PREPROCESSING PATHS
#------------------------------------------------------------------------------
DATA_PATH = '/deepstore/datasets/hmi/speechlaugh-corpus/ami' # path to AMI Meeting Corpus in deepstore
# DATA_PATH = '../data' # path to AMI Meeting Corpus in local

TRANS_PATH = os.path.join(DATA_PATH, 'transcripts')
TRANS_SEG_PATH = os.path.join(DATA_PATH, 'transcript_segments')

SOURCE_PATH = os.path.join(DATA_PATH, 'amicorpus')
AUDIO_PATH = os.path.join(DATA_PATH, 'audio_segments')
VIDEO_PATH = os.path.join(DATA_PATH, 'video_segments')

# PATH TO STORE DISFLUENCY/LAUGHTER MARKERS + AUDIO/VIDEO SEGMENTS
DSFL_PATH = os.path.join(DATA_PATH, 'dsfl_laugh')
DSFL_CSV_PATH = os.path.join(DSFL_PATH, 'disfluency_laughter_markers.csv')

DATASET_PATH = os.path.join(DATA_PATH, 'dataset')

#------------------------------------------------------------------------------
# DETECTORS PATHS
#------------------------------------------------------------------------------
DETECTORS_PATH = '../resources'
FACE_PREDICTOR_PATH = os.path.join(DETECTORS_PATH, 'shape_predictor_68_face_landmarks.dat')
CNN_DETECTOR_PATH = os.path.join(DETECTORS_PATH, 'mmod_human_face_detector.dat')
MEAN_FACE_PATH = os.path.join(DETECTORS_PATH, '20words_mean_face.npy')

#------------------------------------------------------------------------------
# MODEL URLS
#------------------------------------------------------------------------------
MODEL_URLS = {
    "mean_face": "https://github.com/speechlaugh/speechlaugh-corpus/releases/download/v0.1/20words_mean_face.npy",
    "face_predictor": "https://github.com/speechlaugh/speechlaugh-corpus/releases/download/v0.1/shape_predictor_68_face_landmarks.dat",
    "cnn_detector": "https://github.com/speechlaugh/speechlaugh-corpus/releases/download/v0.1/mmod_human_face_detector.dat"
}
