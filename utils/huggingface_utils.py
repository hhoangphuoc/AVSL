from datasets import Dataset, Audio, Video, load_from_disk
import math # Added for isnan
import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder, upload_large_folder
import pandas as pd
import time
import random
from requests.exceptions import HTTPError
import re

def av_to_hf_dataset(recordings, dataset_path=None, prefix="ami"):
    """
    Create a HuggingFace dataset from the processed segments (audio, video, and lip videos), 
    along with the transcript text.
    This function will create a HuggingFace Dataset with AudioFolder(`audio`) and VideoFolder(`video` and `lip_video`) as well.
    The folder should include `metadata.jsonl` at the root scope, which points to the audio, video, and lip_video files in corresponding folders.
    The folder structure should be as follows:

    dataset_path/
        data/
            audio_file (e.g. ES2002a-0.00-0.10-audio.wav)
            video_file (e.g. ES2002a-0.00-0.10-video.mp4)
            lip_video_file (e.g. ES2002a-0.00-0.10-lip_video.mp4)
        metadata.jsonl
        {prefix}-segments-info.csv
        ...

    Args:
        recordings: List of dictionaries containing segment information
        dataset_path: Path to HuggingFace Dataset. If None, defaults to `DATA_PATH/dataset`
        prefix: Prefix for the dataset name.
    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create the data directory for proper Huggingface VideoFolder format
    data_dir = os.path.join(dataset_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # GENERATE THE DATASET FROM THE RECORDINGS
    df = pd.DataFrame(recordings)

    # save the dataframe to a csv file
    csv_path = os.path.join(dataset_path, f'{prefix}-segments-info.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Save metadata as JSON for easier handling during upload
    metadata_path = os.path.join(dataset_path, f'metadata.jsonl')
    print(f"Saving metadata to jsonl file: {metadata_path}")

    # Process each recording in batches to avoid excessive output
    batch_size = 100
    num_batches = (len(recordings) + batch_size - 1) // batch_size
    
    with open(metadata_path, 'w') as f:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(recordings))
            batch = recordings[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})...")
            
            for record in batch:
                # Create a copy of the record to avoid modifying the original
                metadata = record.copy()

                # Copy audio file to data directory
                if 'audio' in metadata and os.path.exists(metadata['audio']):
                    audio_file = os.path.basename(metadata['audio'])
                    destination_path = os.path.join(data_dir, audio_file)
                    if metadata['audio'] != destination_path and not os.path.exists(destination_path):
                        shutil.copy(metadata['audio'], destination_path)
                    metadata['audio'] = f"data/{audio_file}"
                
                # Copy video file to data directory
                if 'video' in metadata and os.path.exists(metadata['video']):
                    video_file = os.path.basename(metadata['video'])
                    destination_path = os.path.join(data_dir, video_file)
                    if metadata['video'] != destination_path and not os.path.exists(destination_path):
                        shutil.copy(metadata['video'], destination_path)
                    metadata['video'] = f"data/{video_file}"
                    
                    # Add format info for video
                    metadata['video_format'] = {'fps': 25.0}

                # Copy lip video file to data directory
                if 'lip_video' in metadata and os.path.exists(metadata['lip_video']):
                    lip_video_file = os.path.basename(metadata['lip_video'])
                    destination_path = os.path.join(data_dir, lip_video_file)
                    if metadata['lip_video'] != destination_path and not os.path.exists(destination_path):
                        shutil.copy(metadata['lip_video'], destination_path)
                    metadata['lip_video'] = f"data/{lip_video_file}"
                    
                    # Add format info for lip_video
                    metadata['lip_video_format'] = {'fps': 25.0}
                
                # Add format info for audio
                if 'audio' in metadata:
                    metadata['audio_format'] = {'sampling_rate': 16000}
                
                # Write the metadata as a JSON line
                f.write(json.dumps(metadata) + '\n')
    
    # Create README.md if it doesn't exist
    readme_path = os.path.join(dataset_path, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(f"# {prefix.upper()} Dataset\n\n")
            f.write(f"This dataset contains segmented audio and video clips.\n\n")
            f.write(f"- Number of recordings: {len(recordings)}\n")
            f.write(f"- Has audio: {'audio' in df.columns}\n")
            f.write(f"- Has video: {'video' in df.columns}\n")
            f.write(f"- Has lip video: {'lip_video' in df.columns}\n\n")
            f.write("## Dataset Format\n\n")
            f.write("This dataset follows the VideoFolder format recommended by Huggingface.\n")
            f.write("All media files are stored in the 'data/' directory and referenced in metadata.jsonl.\n")
    
    # Create HuggingFace Dataset containing all recordings
    dataset = Dataset.from_pandas(df)
    
    # Add audio and video features
    if 'audio' in dataset.features:
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    if 'video' in dataset.features:
        dataset = dataset.cast_column('video', Video())
        
    if 'lip_video' in dataset.features:
        dataset = dataset.cast_column('lip_video', Video())
    
    # Save the dataset
    print(f"\nSaving dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    print(f"HuggingFace dataset saved: {dataset}")
    
    print("\nIMPORTANT: To upload this dataset to the HuggingFace Hub:")
    print("1. Use the `push_dataset_to_hub` function from this module")
    print("2. Make sure you have sufficient API rate limits with HuggingFace")
    print("3. Consider incrementally uploading if you have a large dataset")


# ================================================================================================================
def push_dataset_to_hub(dataset_path, repo_name, token=None, private=True, max_retries=5, initial_delay=5):
    """
    Push the dataset to the HuggingFace Hub with proper authentication and configuration.
    This function handles datasets with video files using upload_large_folder for reliable uploading.
    Implements exponential backoff to handle rate limiting.
    
    Args:
        dataset_path: Path to the HuggingFace dataset
        repo_name: Name of the repository (format: 'dataset-name')
        token: HuggingFace API token. If None, will use the token from huggingface-cli login
        private: Whether to create a private repository (default: True)
        max_retries: Maximum number of retries for rate-limited operations
        initial_delay: Initial delay in seconds before retrying
    
    Returns:
        None
    """
    # Helper function for exponential backoff
    def retry_with_exponential_backoff(operation_func, max_retries=max_retries, initial_delay=initial_delay):
        """Retry an operation with exponential backoff."""
        retries = 0
        while True:
            try:
                return operation_func()
            except Exception as e:
                if retries >= max_retries:
                    print(f"Maximum retries ({max_retries}) exceeded. Last error: {e}")
                    raise
                    
                # Check if it's a rate limiting error
                if isinstance(e, HTTPError) and "429" in str(e):
                    # Extract wait time if available in the message
                    wait_time_match = re.search(r"retry this action in (\d+) minutes", str(e))
                    if wait_time_match:
                        wait_mins = int(wait_time_match.group(1))
                        wait_secs = wait_mins * 60
                        print(f"Rate limited. Waiting for {wait_mins} minutes as instructed by the API...")
                        time.sleep(wait_secs + random.uniform(1, 5))  # Add a small random delay
                    else:
                        # Calculate exponential backoff with jitter
                        wait_time = (2 ** retries) * initial_delay + random.uniform(0, 1)
                        print(f"Rate limited. Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    
                    retries += 1
                    continue
                else:
                    # For other errors, raise immediately
                    raise
    
    try:
        # Convert private to boolean if it's a string
        if isinstance(private, str):
            private = private.lower() == 'true'
        
        # Step 1: Initialize the Hugging Face API
        api = HfApi(token=token)
        
        # Step 2: Create the repository if it doesn't exist
        repo_id = f"hhoangphuoc/{repo_name}"
        
        # Wrap repo creation in retry logic
        def create_repo_with_retry():
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    token=token
                )
                print(f"Created new dataset repository: {repo_id}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"Repository {repo_id} already exists")
                else:
                    raise
        
        retry_with_exponential_backoff(create_repo_with_retry)
        
        # Ensure dataset_path is a string path
        if not isinstance(dataset_path, str):
            if hasattr(dataset_path, 'cache_files') and dataset_path.cache_files:
                dataset_path = dataset_path.cache_files[0]['filename'].rsplit('/', 1)[0]
            else:
                raise ValueError("Dataset must be a path or a Dataset with cache_files")
        
        print(f"Preparing dataset from: {dataset_path}")
        
        # Check if this is a video dataset by looking at the metadata.jsonl
        metadata_path = os.path.join(dataset_path, "metadata.jsonl")
        is_video_dataset = False
        
        if os.path.exists(metadata_path):
            # Read the first line to check for video paths
            with open(metadata_path, 'r') as f:
                first_record = json.loads(f.readline().strip())
                is_video_dataset = 'video' in first_record or 'lip_video' in first_record
        
        # Optimize upload strategy based on dataset type
        if is_video_dataset:
            print("Dataset contains video files. Using optimized video upload approach...")
            
            # First verify the dataset structure
            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                # Create data dir if it doesn't exist
                os.makedirs(data_dir, exist_ok=True)
                
                # Check if we have audio/video/lips folders instead
                for folder in ['audio', 'video', 'lips']:
                    folder_path = os.path.join(dataset_path, folder)
                    if os.path.exists(folder_path):
                        print(f"Found {folder} folder. Moving files to data/ directory...")
                        for file in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, file)
                            if os.path.isfile(file_path):
                                shutil.copy2(file_path, os.path.join(data_dir, file))
            
            # Use a single upload approach with retry
            def upload_dataset_with_retry():
                print(f"Uploading dataset to {repo_id} using optimized large folder upload...")
                upload_large_folder(
                    folder_path=dataset_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    ignore_patterns=["*.gitignore", "upload_tmp/*"],
                )
                print(f"Successfully uploaded dataset to {repo_id}")
            
            retry_with_exponential_backoff(upload_dataset_with_retry)
            
        else:
            # For non-video datasets, use push_to_hub
            try:
                dataset = load_from_disk(dataset_path)
                
                def push_dataset_with_retry():
                    print("Dataset does not contain video files. Using standard push_to_hub.")
                    dataset.push_to_hub(
                        repo_id=repo_id,
                        private=private,
                        token=token,
                        embed_external_files=True,
                        max_shard_size="500MB"
                    )
                    print(f"Successfully uploaded dataset to {repo_id}")
                
                retry_with_exponential_backoff(push_dataset_with_retry)
                
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Falling back to direct upload...")
                
                def upload_fallback_with_retry():
                    upload_large_folder(
                        folder_path=dataset_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        ignore_patterns=["*.gitignore", "upload_tmp/*"],
                    )
                    print(f"Successfully uploaded dataset to {repo_id} using fallback method")
                
                retry_with_exponential_backoff(upload_fallback_with_retry)
        
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Error pushing dataset to hub: {e}")
        raise e


# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push the dataset to the HuggingFace Hub')
    parser.add_argument('--dataset_path', type=str, default='../data/dsfl/dataset', help='Path to the HuggingFace dataset')
    parser.add_argument('--repo_name', type=str, default='ami-disfluency-AV', help='Name of the repository to push the dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--private', default=False, help='Whether to create a private repository')
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    push_dataset_to_hub(args.dataset_path, args.repo_name, args.token, args.private)

