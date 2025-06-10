# Laughter Dataset Processing SBATCH Script

This document describes how to use the `laugh_dataset_process.sh` SBATCH script to process the AMI laughter dataset on an HPC cluster.

## Overview

The script performs the following steps:
1. Sets up the environment and loads required modules
2. Activates the `whisper-flamingo` conda environment
3. Verifies all dependencies are installed
4. Runs tests to ensure the dataset is valid
5. Processes the laughter dataset with optional parallel processing

## Basic Usage

Submit the job with default settings (standard processing):
```bash
sbatch laugh_dataset_process.sh
```

## Advanced Usage

### Using Chunked Processing (Recommended for Large Datasets)

For processing large datasets with better memory management and checkpoint support:
```bash
sbatch laugh_dataset_process.sh --chunked
```

### Custom Chunk Size
```bash
sbatch laugh_dataset_process.sh --chunked --chunk_size=500
```

### Adjust Worker Processes
```bash
sbatch laugh_dataset_process.sh --chunked --workers=16
```

### Adjust Batch Size for Frame Processing
```bash
sbatch laugh_dataset_process.sh --chunked --batch_size=32
```

### Complete Example with All Options
```bash
sbatch laugh_dataset_process.sh --chunked --chunk_size=1000 --workers=12 --batch_size=16
```

## Processing Modes

### Standard Mode (Default)
- Processes the entire dataset in one run
- Suitable for smaller datasets (<10,000 segments)
- No checkpoint support
- Uses the `laugh_dataset_process.py` script directly

### Chunked Mode (`--chunked`)
- Divides the dataset into smaller chunks
- Processes each chunk independently
- Saves progress after each chunk
- Can resume from failures
- Better memory management
- Supports parallel lip video extraction
- Automatically merges chunks at the end

## Resource Requirements

The script requests:
- **CPUs**: 16 cores
- **Memory**: 64GB
- **GPU**: 1 NVIDIA Ampere GPU (A40)
- **Time**: 48 hours

Adjust these in the SBATCH headers if needed for your dataset size.

## Output Structure

The processing creates the following structure:

```
/home/s2587130/AVSL/data/ami_laugh/
├── segments/
│   ├── audio_segments/        # Extracted audio clips
│   ├── video_segments/        # Extracted video clips
│   ├── lip_videos/           # Extracted lip region videos
│   └── dataset_records.json  # Metadata for all segments
└── dataset/
    ├── data/                 # HuggingFace dataset files
    ├── dataset_dict.json     # Dataset structure
    └── metadata.jsonl        # Segment metadata
```

For chunked processing, intermediate chunk directories are created:
```
segments/
├── chunk_0000/
├── chunk_0001/
├── ...
```

## Monitoring Progress

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Output
```bash
tail -f logs/laugh_process-<JOB_ID>.out
```

### Check Error Log
```bash
tail -f logs/laugh_process-<JOB_ID>.err
```

## Troubleshooting

### Common Issues

1. **Dependency Installation Failures**
   - The script attempts to install missing dependencies automatically
   - If dlib installation fails, consider pre-installing it in the conda environment
   - Check the dependency test output in the log file

2. **Memory Issues**
   - Use chunked processing with smaller chunk sizes
   - Reduce the batch size for frame processing
   - Request more memory in the SBATCH header

3. **GPU Issues**
   - Ensure CUDA modules are loaded correctly
   - Check GPU availability with `nvidia-smi`
   - The script can fall back to CPU processing if needed

4. **Time Limit Exceeded**
   - Use chunked processing to save progress
   - Process can be resumed by resubmitting the job
   - Increase time limit in SBATCH header if needed

### Resuming Failed Jobs

If using chunked processing and the job fails:
1. Check which chunks were completed in the status files
2. The script will automatically skip completed chunks
3. Simply resubmit the job to continue processing

## Performance Tips

1. **Optimal Chunk Size**: 500-1000 segments per chunk
2. **Worker Processes**: Set to number of CPU cores minus 2-4
3. **Batch Size**: 16-32 for GPU processing, 8-16 for CPU
4. **Parallel Processing**: Automatically enabled for >10 videos

## Example Workflow

1. First, generate the laughter markers CSV:
   ```bash
   python disfluency_laughter_process.py
   ```

2. Submit the processing job:
   ```bash
   sbatch laugh_dataset_process.sh --chunked --chunk_size=1000
   ```

3. Monitor progress:
   ```bash
   watch -n 60 'tail -20 logs/laugh_process-*.out'
   ```

4. Check results:
   ```bash
   python -c "
   import json
   with open('path/to/your/directory/data/ami_laughter/segments/dataset_records.json') as f:
       records = json.load(f)
   print(f'Total segments: {len(records)}')
   "
   ```

## Integration with Model Training

The generated dataset can be loaded using:
```python
from datasets import load_from_disk

dataset = load_from_disk('path/to/your/directory/data/ami_laughter/dataset')
``` 