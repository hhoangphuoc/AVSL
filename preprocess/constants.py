import os


DATA_PATH = '/deepstore/datasets/hmi/speechlaugh-corpus/ami' # path to AMI Meeting Corpus in deepstore
# DATA_PATH = '../data' # path to AMI Meeting Corpus in local

TRANS_PATH = os.path.join(DATA_PATH, 'transcripts')
TRANS_SEG_PATH = os.path.join(DATA_PATH, 'transcript_segments')

SOURCE_PATH = os.path.join(DATA_PATH, 'amicorpus')
AUDIO_PATH = os.path.join(DATA_PATH, 'audio_segments')
VIDEO_PATH = os.path.join(DATA_PATH, 'video_segments')

DATASET_PATH = os.path.join(DATA_PATH, 'dataset')


