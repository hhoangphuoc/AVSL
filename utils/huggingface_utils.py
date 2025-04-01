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

def av_to_hf_dataset(recordings, dataset_path=None):
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

    # save the dataframe to a csv file
    csv_path = os.path.join(dataset_path, 'ami-segmented-recordings.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    
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
    
    # Add audio and video features
    if 'audio' in dataset.features:
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    if 'video' in dataset.features:
        dataset = dataset.cast_column('video', Video())
        
    if 'lip_video' in dataset.features:
        dataset = dataset.cast_column('lip_video', Video())
    
    # Save the dataset
    print(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    print(f"HuggingFace dataset saved: {dataset}")


# ================================================================================================================
def push_dataset_to_hub(dataset_path, repo_name, token=None, private=True):
    """
    Push the dataset to the HuggingFace Hub with proper authentication and configuration.
    This function handles datasets with video files using upload_large_folder for reliable uploading.
    
    Args:
        dataset_path: Path to the HuggingFace dataset
        repo_name: Name of the repository (format: 'dataset-name')
        token: HuggingFace API token. If None, will use the token from huggingface-cli login
        private: Whether to create a private repository (default: True)
    
    Returns:
        None
    """
    try:
        # Convert private to boolean if it's a string
        if isinstance(private, str):
            private = private.lower() == 'true'
        
        # Step 1: Initialize the Hugging Face API
        api = HfApi(token=token)
        
        # Step 2: Create the repository if it doesn't exist
        repo_id = f"hhoangphuoc/{repo_name}"
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=token
            )
            print(f"Created new dataset repository: {repo_id}")
        except Exception as e:
            print(f"Repository {repo_id} already exists")
        
        # Ensure dataset_path is a string path
        if not isinstance(dataset_path, str):
            if hasattr(dataset_path, 'cache_files') and dataset_path.cache_files:
                dataset_path = dataset_path.cache_files[0]['filename'].rsplit('/', 1)[0]
            else:
                raise ValueError("Dataset must be a path or a Dataset with cache_files")
        
        print(f"Preparing dataset from: {dataset_path}")
        
        # First, load the dataset to check if it has video files
        try:
            dataset = load_from_disk(dataset_path)
            has_video = 'video' in dataset.features or 'lip_video' in dataset.features
            
            # Case 1: Dataset with no videos - use push_to_hub
            if not has_video:
                print("Dataset does not contain video files. Using standard push_to_hub.")
                dataset.push_to_hub(
                    repo_id=repo_id,
                    private=private,
                    token=token,
                    embed_external_files=True,
                    max_shard_size="500MB"
                )
            
            # Case 2: Dataset with videos - use upload_large_folder
            else:
                print("Dataset contains video files. Using upload_large_folder approach.")
                

                
                # Use upload_large_folder to push the dataset with video content
                print(f"Uploading dataset to {repo_id} using upload_large_folder...")
                upload_large_folder(
                    folder_path=dataset_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    ignore_patterns=["*.gitignore", "upload_tmp/*"],
                )
                
            print(f"Successfully uploaded dataset to {repo_id}")
            print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
                
        except Exception as specific_error:
            print(f"Error while processing dataset: {specific_error}")
            print("Attempting direct folder upload as fallback...")
            
            # Fallback: Direct upload of the folder using upload_large_folder
            print(f"Fallback: Uploading dataset to {repo_id} using upload_large_folder...")
            upload_large_folder(
                folder_path=dataset_path,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=["*.gitignore", "upload_tmp/*"],
            )
            print(f"Successfully uploaded dataset to {repo_id} using fallback method")
        
    except Exception as e:
        print(f"Error pushing dataset to hub: {e}")
        raise e


# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push the dataset to the HuggingFace Hub')
    parser.add_argument('--dataset_path', type=str, default='../data/ami_dataset', help='Path to the HuggingFace dataset')
    parser.add_argument('--repo_name', type=str, default='ami-av', help='Name of the repository to push the dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--private', default=True, help='Whether to create a private repository')
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    push_dataset_to_hub(args.dataset_path, args.repo_name, args.token, args.private)

