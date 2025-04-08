from datasets import Dataset, Audio, Video
import os
import shutil
import json
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_large_folder
import pandas as pd
import time
import random
from requests.exceptions import HTTPError
import re

def av_to_hf_dataset(recordings, dataset_path=None, prefix="ami"):
    """
    Create a HuggingFace dataset from the processed segments (audio, video, and lip videos), 
    along with the transcript text.
    
    Args:
        recordings: List of dictionaries containing segment information
        dataset_path: Path to HuggingFace Dataset. If None, defaults to `DATA_PATH/dataset`
    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # GENERATE THE DATASET FROM THE RECORDINGS
    df = pd.DataFrame(recordings)
    
    # Save metadata as JSON for easier handling during upload
    metadata_path = os.path.join(dataset_path, 'metadata.jsonl')
    with open(metadata_path, 'w') as f:
        for record in recordings:
            # Create a copy of the record to avoid modifying the original
            metadata = record.copy()
            
            # Store relative paths instead of absolute paths
            if 'audio' in metadata:
                metadata['audio'] = os.path.basename(metadata['audio'])
            if 'video' in metadata:
                metadata['video'] = os.path.basename(metadata['video'])
            if 'lip_video' in metadata:
                metadata['lip_video'] = os.path.basename(metadata['lip_video'])
                
            # Write the metadata as a JSON line
            f.write(json.dumps(metadata) + '\n')
    
    # Create HuggingFace Dataset containing all recordings
    dataset = Dataset.from_pandas(df)
    
    # Cast audio and video features to HuggingFace Audio and Video
    # NOTE: Dataset features casting required full path
    if 'audio' in dataset.features: 
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    if 'video' in dataset.features:
        dataset = dataset.cast_column('video', Video())
        
    if 'lip_video' in dataset.features:
        dataset = dataset.cast_column('lip_video', Video())

    # save the dataframe to a csv file
    # NOTE: The CSV path is relative to the dataset_path, will be: 'data/...` instead of full path
    df['audio'] = df['audio'].apply(lambda x: os.path.join('data', os.path.basename(x)))
    df['video'] = df['video'].apply(lambda x: os.path.join('data', os.path.basename(x)))
    df['lip_video'] = df['lip_video'].apply(lambda x: os.path.join('data', os.path.basename(x)))
    csv_path = os.path.join(dataset_path, f'{prefix}-segmented-info.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Save the dataset
    print(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    print(f"HuggingFace dataset saved: {dataset}")

#====================================================================================================

def av_to_hf_dataset_with_shards(recordings, dataset_path=None, prefix="ami", files_per_shard=5000):
    """
    Create a HuggingFace dataset from processed segments (audio, video, lip videos)
    Stores metadata in Arrow/Parquet format and media files in multiple shards of 'data/' directories.
    
    NOTE: This function is designed for uploading large dataset with both audio, video, and lip video clips 
    to the HuggingFace Hub, for better DataViewer. For normal use, you can use `av_to_hf_dataset` instead.

    Args:
        recordings: List of dictionaries containing segment information.
        dataset_path: Path to save the HuggingFace dataset structure.
        prefix: Prefix for dataset-related files (e.g., CSV info file).
        files_per_shard: Target number of media files per shard.
    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    if not dataset_path:
        raise ValueError("dataset_path must be provided")
        
    dataset_path_abs = os.path.abspath(dataset_path) # Get absolute path for reference
    print(f"Dataset path: {dataset_path_abs}")
    os.makedirs(dataset_path_abs, exist_ok=True)
    
    # --- 1. Prepare DataFrame with correct relative paths --- 
    
    # Estimate total number of media files for sharding calculation
    estimated_media_files = sum(
        int(os.path.exists(record.get('audio', ''))) + 
        int(os.path.exists(record.get('video', ''))) + 
        int(os.path.exists(record.get('lip_video', ''))) 
        for record in recordings
    )
    num_shards = max(1, (estimated_media_files + files_per_shard - 1) // files_per_shard)
    print(f"Estimated {estimated_media_files} media files. Planning for {num_shards} shards.")
    
    processed_records = []
    files_copied_count = 0
    media_files_in_shards = [0] * num_shards
    
    print("Processing recordings and copying media files to sharded directories...")
    for idx, record in enumerate(tqdm(recordings, desc="Processing records")):
        shard_idx = idx % num_shards
        shard_name = f"shard_{shard_idx:04d}"
        shard_dir_abs = os.path.join(dataset_path_abs, "data", shard_name) #dataset_path/data/shard_0000/
        os.makedirs(shard_dir_abs, exist_ok=True)
        
        metadata = record.copy()
        current_record_media_count = 0
        
        # ------------------------ Process and copy audio ----------------------------------------------
        if 'audio' in metadata and metadata['audio'] and os.path.exists(metadata['audio']):
            try:
                audio_file = os.path.basename(metadata['audio'])
                destination_path = os.path.join(shard_dir_abs, audio_file)
                if not os.path.exists(destination_path):
                    shutil.copy2(metadata['audio'], destination_path)
                    files_copied_count += 1
                    current_record_media_count += 1
                metadata['audio'] = destination_path #metadata['audio'] = dataset_path/data/shard_0000/audio_file.wav
            except Exception as e:
                print(f"\nWarning: Failed to copy audio {metadata['audio']} for record {idx}: {e}")
                metadata['audio'] = None
        elif 'audio' in metadata:
             metadata['audio'] = None
        # ------------------------------------------------------------------------------------------------
             

        # ------------------------ Process and copy video ------------------------------------------------
        if 'video' in metadata and metadata['video'] and os.path.exists(metadata['video']):
            try:
                video_file = os.path.basename(metadata['video'])
                destination_path = os.path.join(shard_dir_abs, video_file)
                if not os.path.exists(destination_path):
                    shutil.copy2(metadata['video'], destination_path)
                    files_copied_count += 1
                    current_record_media_count += 1

                metadata['video'] = destination_path
            except Exception as e:
                 print(f"\nWarning: Failed to copy video {metadata['video']} for record {idx}: {e}")
                 metadata['video'] = None
        elif 'video' in metadata:
             metadata['video'] = None
        # ------------------------------------------------------------------------------------------------


        # ------------------------ Process and copy lip_video -------------------------------------------
        if 'lip_video' in metadata and metadata['lip_video'] and os.path.exists(metadata['lip_video']):
             try:
                lip_video_file = os.path.basename(metadata['lip_video'])
                destination_path = os.path.join(shard_dir_abs, lip_video_file)
                if not os.path.exists(destination_path):
                    shutil.copy2(metadata['lip_video'], destination_path)
                    files_copied_count += 1
                    current_record_media_count += 1
                metadata['lip_video'] = destination_path #metadata['lip_video'] = dataset_path/data/shard_0000/lip_video_file.mp4
             except Exception as e:
                 print(f"\nWarning: Failed to copy lip_video {metadata['lip_video']} for record {idx}: {e}")
                 metadata['lip_video'] = None
        elif 'lip_video' in metadata:
             metadata['lip_video'] = None
        # ------------------------------------------------------------------------------------------------

        media_files_in_shards[shard_idx] += current_record_media_count
        processed_records.append(metadata)
        
    print(f"Finished processing. Copied {files_copied_count} media files into {num_shards} shards.")

    print("---------------------------------------------")
    print("Shard distribution:")
    for i, count in enumerate(media_files_in_shards):
         print(f"  Shard {i:04d}: {count} files")
    print("---------------------------------------------")

    # --- Create HuggingFace Dataset --- 
    df = pd.DataFrame(processed_records)

    # Save the metadata to a jsonl file ----------------------------------------------------------
    metadata_path = os.path.join(dataset_path_abs, f'metadata.jsonl')
    print(f"Saving metadata to jsonl file: {metadata_path}")
    with open(metadata_path, 'w') as f:
        for record in processed_records:
            f.write(json.dumps(record) + '\n')
    #--------------------------------------------------------------------------------------------

    # Save the dataframe to a csv file ----------------------------------------------------------------
    csv_path = os.path.join(dataset_path_abs, f'{prefix}-segmented-info.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    #--------------------------------------------------------------------------------------------

    
    # ---------------------------------------- USING STANDARD HF DATASET API ---------------------------
    print("Creating HuggingFace Dataset (with relative paths)...")
    try:
        dataset = Dataset.from_pandas(df)
        
        # Cast features AFTER the dataset has relative paths already
        print("Casting features...")
        if 'audio' in dataset.features:
            try:
                # For audio, use absolute paths only during validation if needed
                dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
            except Exception as e:
                print(f"Warning: Could not cast audio column: {e}")
                
        if 'video' in dataset.features:
            try:
                # Disable decoding for videos
                dataset = dataset.cast_column('video', Video())
            except Exception as e:
                print(f"Warning: Could not cast video column: {e}")
                
        if 'lip_video' in dataset.features:
            try:
                # Disable decoding for lip videos
                dataset = dataset.cast_column('lip_video', Video())
            except Exception as e:
                print(f"Warning: Could not cast lip_video column: {e}")
                
        # Save the dataset with relative paths
        try:
            print(f"Saving dataset structure to {dataset_path_abs}...")
            dataset.save_to_disk(dataset_path_abs)
            print(f"HuggingFace dataset structure saved successfully.")
        except Exception as e:
            print(f"\nFATAL ERROR saving dataset structure: {e}")
            raise
            
    except Exception as e:
        print(f"Error creating Dataset: {e}")
    
    #------------------------------------------------------------------------------------------------
    print("\nDataset preparation complete.")
    print(f"Dataset structure saved at: {dataset_path_abs}")
    print("Ensure this entire directory (including the `data` folder with shards) is uploaded to the Hub.")


# ================================================================================================================
def push_dataset_to_hub(dataset_path, repo_name, token=None, private=True, max_retries=5, initial_delay=5):
    """
    Push the dataset (metadata + sharded media files) to the HuggingFace Hub.
    Uses upload_large_folder and implements exponential backoff.
    
    Args:
        dataset_path: Path to the locally saved HuggingFace dataset.
        repo_name: Name of the repository (format: 'dataset-name').
        token: HuggingFace API token.
        private: Whether to create a private repository.
        max_retries: Max retries for rate-limited operations.
        initial_delay: Initial delay for backoff.
    
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
                is_rate_limit = False
                if isinstance(e, HTTPError) and e.response is not None and e.response.status_code == 429:
                    is_rate_limit = True
                    
                # Also check for specific string if HTTPError isn't raised directly but message exists
                # (Sometimes the hub client might wrap the error)
                if not is_rate_limit and "rate-limited" in str(e).lower():
                     is_rate_limit = True

                if is_rate_limit:
                    # Extract wait time if available in the message
                    wait_time_match = re.search(r"retry.*?in (\d+) minutes", str(e), re.IGNORECASE)
                    wait_time_match_secs = re.search(r"retry.*?in (\d+) seconds", str(e), re.IGNORECASE)
                    
                    wait_secs = 0
                    if wait_time_match:
                        wait_mins = int(wait_time_match.group(1))
                        wait_secs = wait_mins * 60
                        print(f"Rate limited. Waiting for {wait_mins} minutes as instructed by the API...")
                    elif wait_time_match_secs:
                        wait_secs = int(wait_time_match_secs.group(1))
                        print(f"Rate limited. Waiting for {wait_secs} seconds as instructed by the API...")
                    else:
                        # Calculate exponential backoff with jitter if no specific time given
                        wait_secs = (2 ** retries) * initial_delay + random.uniform(0, 1)
                        print(f"Rate limited. Retrying in {wait_secs:.2f} seconds (exponential backoff)...")
                    
                    time.sleep(max(1.0, wait_secs) + random.uniform(0, 2)) # Ensure minimum wait, add jitter
                    retries += 1
                    continue
                else:
                    # For other errors, raise immediately
                    print(f"Non-rate-limit error encountered: {type(e).__name__}")
                    raise e # Raise the original error
            
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
                # Check for specific Huggingface Hub error for existing repo
                if isinstance(e, HTTPError) and e.response is not None and e.response.status_code == 409:
                     print(f"Repository {repo_id} already exists (HTTP 409).")
                elif "already exists" in str(e).lower(): # Fallback check
                     print(f"Repository {repo_id} already exists.")
                else:
                     print(f"Error creating repo (will retry if possible): {e}")
                     raise # Re-raise to trigger retry or fail
        
        print(f"Ensuring repository {repo_id} exists...")
        retry_with_exponential_backoff(create_repo_with_retry)
        
        # Ensure dataset_path is a string path
        if not isinstance(dataset_path, str):
             raise ValueError("dataset_path must be a string")
        if not os.path.isdir(dataset_path):
             raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
             
        print(f"Preparing to upload dataset from: {dataset_path}")
        
        # --- Determine Upload Strategy --- 
        # No need to explicitly check for video anymore. 
        # upload_large_folder is suitable for the structure created by av_to_hf_dataset
        # (config files + data/shards/media_files)
        print("Using upload_large_folder strategy for dataset with sharded media.")

        # --- Upload the entire dataset folder (config + data shards) --- 
        def upload_dataset_with_retry():
            print(f"Uploading dataset folder to {repo_id} using upload_large_folder...")
            upload_large_folder(
                folder_path=dataset_path,
                repo_id=repo_id,
                repo_type="dataset",
                # ignore_patterns=["*.gitignore", "upload_tmp/*", f"{prefix}-segments-info.csv"], # Optionally ignore the extra CSV
                ignore_patterns=["*.gitignore", "upload_tmp/*"], 
                # Consider adding commit_message if desired
                # commit_message=f"Upload dataset from {datetime.now().isoformat()}"
            )
            print(f"Successfully uploaded dataset folder to {repo_id}")
        
        retry_with_exponential_backoff(upload_dataset_with_retry)
        
        print(f"\nUpload complete.")
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        print("Note: It might take a few minutes for the Dataset Viewer to process the data.")
        
    except Exception as e:
        print(f"\n---------------------")
        print(f"FATAL ERROR pushing dataset to hub: {e}")
        print(f"Type: {type(e).__name__}")
        # If it's an HTTPError, print more details
        if isinstance(e, HTTPError) and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            try:
                 print(f"Response Content: {e.response.text}")
            except Exception:
                 print("Could not read response content.")
        print(f"---------------------")
        raise e


# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push the dataset to the HuggingFace Hub')
    parser.add_argument('--dataset_path', type=str, default='/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/dataset', help='Path to the HuggingFace dataset')
    parser.add_argument('--repo_name', type=str, default='ami-disfluency', help='Name of the repository to push the dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--private', default=False, help='Whether to create a private repository')
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    push_dataset_to_hub(args.dataset_path, args.repo_name, args.token, args.private)

