# THIS FILE IS USED TO CHECK THE STATS INFORMATION OF SEGMENTED DATASET, STORED IN DEEPSTORE


# ------------------------------------------------------------------------------------------------
#                       FOR ORIGINAL AUDIO/VIDEO SEGMENTED DATASET
#               Path: /deepstore/datasets/hmi/speechlaugh-corpus/ami/
# ------------------------------------------------------------------------------------------------

# Checking size of audio/video segments directories
# OUTPUT:
# Files count: 80,285
# Total size: 13G
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/audio_segments -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/audio_segments | cut -f1)"

# ------------------------------------------------------------------------------------------------

# Checking size of video segments directory
# OUTPUT:

# Original video segments
# Files count: 78,685
# Total size: 5.6G
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/original_videos -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/original_videos | cut -f1)"

# Lip video segments
# Files count: 67,438
# Total size: 1.9G
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lips -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lips | cut -f1)"

# ================================================================================================================  

# ------------------------------------------------------------------------------------------------
#                       FOR DISFLUENCY/LAUGHTER ONLY SEGMENTED DATASET
#               Path: /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl_laugh/
# ------------------------------------------------------------------------------------------------

# Checking total entries, laughter entries, and disfluency entries of the disfluency_laughter_markers.csv file
echo "Total entries:" && 
wc -l dsfl_laugh/disfluency_laughter_markers.csv && 
echo "Laugh entries:" && 
grep -c ,,1 dsfl_laugh/disfluency_laughter_markers.csv && 
echo "Disfluency entries:" && 
grep -v ,,1 dsfl_laugh/disfluency_laughter_markers.csv | 
grep -v disfluency_type |   
wc -l

# ------------------------------------------------------------------------------------------------

# Checking size of disfluency/laughter audio segments directory
# OUTPUT:
# Files count: 52,991
# Total size: 895M
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/audio_segments -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/audio_segments | cut -f1)"

# ------------------------------------------------------------------------------------------------

# Checking size of disfluency/laughter video segments directory
# OUTPUT:
# Files count: 52,542
# Total size: 701M
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/video_segments/dsfl_original -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/video_segments/dsfl_original | cut -f1)"

# ------------------------------------------------------------------------------------------------

# Checking size of disfluency/laughter lip video segments directory
# OUTPUT:
# Files count: 28,263
# Total size: 104M
echo "Files count: $(find /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/video_segments/dsfl_lips -type f | wc -l)"; 
echo "Total size: $(du -sh /deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/video_segments/dsfl_lips | cut -f1)"


