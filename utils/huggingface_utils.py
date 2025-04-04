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

def av_to_hf_dataset(recordings, dataset_path=None, prefix="ami"):
    """
    Create a HuggingFace dataset from the processed segments (audio, video, and lip videos), 
    along with the transcript text.
    This function will create a HuggingFace Dataset with AudioFolder(`audio`) and VideoFolder(`video` and `lip_video`) as well.
    The folder should include `metadata.jsonl` at the root scope, which points to the audio, video, and lip_video files in corresponding folders.
    The folder structure should be as follows:

    dataset_path/
        audio/
            audio_file (e.g. ES2002a-0.00-0.10-audio.wav)
        video/
            video_file (e.g. ES2002a-0.00-0.10-video.mp4)
        lip_video/
            lip_video_file (e.g. ES2002a-0.00-0.10-lip_video.mp4)
        metadata.jsonl
        {prefix}-segments-info.csv
        ...

    Args:
        recordings: List of dictionaries containing segment information
        dataset_path: Path to HuggingFace Dataset. If None, defaults to `DATA_PATH/dataset`
        prefix: Prefix for the dataset name. If None, defaults to `ami`

    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # GENERATE THE DATASET FROM THE RECORDINGS
    df = pd.DataFrame(recordings)

    # save the dataframe to a csv file
    csv_path = os.path.join(dataset_path, f'{prefix}-segments-info.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Save metadata as JSON for easier handling during upload
    metadata_path = os.path.join(dataset_path, f'metadata.jsonl')
    print(f"Saving metadata to jsonl file: {metadata_path}")

    audio_folder = os.path.join(dataset_path, 'audio')
    video_folder = os.path.join(dataset_path, 'video')
    lip_video_folder = os.path.join(dataset_path, 'lips')
    with open(metadata_path, 'w') as f:
        for record in tqdm(recordings, desc="Processing recordings..."):
            # Create a copy of the record to avoid modifying the original
            metadata = record.copy()

            #---------------------- COPY AUDIO, VIDEO, AND LIP VIDEO FILES TO THE CORRESPONDING FOLDERS ----------------------
            
            # Copy audio file to AudioFolder
            if 'audio' in metadata and os.path.exists(metadata['audio']):
                print(f"Copying audio file to AudioFolder: {audio_folder}")
                audio_file = os.path.basename(metadata['audio']) # only the filename, e.g. ES2002a-0.00-0.10-audio.wav
                destination_path = os.path.join(audio_folder, audio_file)
                if metadata['audio'] != destination_path:
                    shutil.copy(metadata['audio'], destination_path)
                metadata['audio'] = f"data/{audio_file}"
            
            # Copy video file to VideoFolder
            if 'video' in metadata and os.path.exists(metadata['video']):
                print(f"Copying video file to VideoFolder: {video_folder}")
                video_file = os.path.basename(metadata['video'])
                destination_path = os.path.join(video_folder, video_file)
                if metadata['video'] != destination_path:
                    shutil.copy(metadata['video'], destination_path)
                metadata['video'] = f"data/{video_file}"

            # Copy lip video file to LipVideoFolder
            if 'lip_video' in metadata and os.path.exists(metadata['lip_video']):
                print(f"Copying lip video file to LipVideoFolder: {lip_video_folder}")
                lip_video_file = os.path.basename(metadata['lip_video'])
                destination_path = os.path.join(lip_video_folder, lip_video_file)
                if metadata['lip_video'] != destination_path:
                    shutil.copy(metadata['lip_video'], destination_path)
                metadata['lip_video'] = f"data/{lip_video_file}"
            
            #--------------------------------------------------------------------------------------------------------------
            
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
    print(f"\nSaving dataset to {dataset_path}")
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

        try:
            dataset = load_from_disk(dataset_path)
            has_video = 'video' in dataset.features or 'lip_video' in dataset.features
            
            # UPLOAD DATASET WITH NO VIDEO FILES
            if not has_video:
                print("Dataset does not contain video files. Using standard push_to_hub.")
                dataset.push_to_hub(
                    repo_id=repo_id,
                    private=private,
                    token=token,
                    embed_external_files=True,
                    max_shard_size="500MB"
                )
                print(f"Successfully uploaded dataset to {repo_id}")

            # UPLOAD DATASET WITH VIDEO FILES (USE `UPLOAD_LARGE_FOLDER`)
            else:
                print("Uploading full dataset content including video files...")
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
            print("Attempting alternative upload approach...")
            
            # Try to load with datasets library as a fallback
            try:
                from datasets import load_dataset
                print("Trying to load dataset instead of using `upload_large_folder`...")
                
                # Check if dataset has a metadata jsonl file
                jsonl_files = [f for f in os.listdir(dataset_path) if f.endswith('.jsonl')]
                if jsonl_files:
                    print(f"Found jsonl files: {jsonl_files}")
                    # Load from jsonl files directly
                    dataset = load_dataset('json', data_files=os.path.join(dataset_path, jsonl_files[0]))

                    # Add audio and video features
                    if 'audio' in dataset.features:
                        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
                    
                    if 'video' in dataset.features:
                        dataset = dataset.cast_column('video', Video())

                    if 'lip_video' in dataset.features:
                        dataset = dataset.cast_column('lip_video', Video())
                        
                    print("Successfully loaded dataset from jsonl. Pushing to hub...")
                    dataset.push_to_hub(
                        repo_id=repo_id,
                        private=private,
                        token=token,
                        embed_external_files=True
                    )
                else:
                    # Fallback to large folder upload
                    print("No jsonl files found. Using direct folder upload...")
                    upload_large_folder(
                        folder_path=dataset_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        ignore_patterns=["*.gitignore", "upload_tmp/*"],
                    )
                print(f"Successfully uploaded dataset to {repo_id}")
            
            except Exception as fallback_error:
                print(f"Fallback approach failed: {fallback_error}")
                print("Attempting direct folder upload as last resort...")
                
                # Fallback: Direct upload of the folder using upload_large_folder
                upload_large_folder(
                    folder_path=dataset_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    ignore_patterns=["*.gitignore", "upload_tmp/*"],
                )
                print(f"Successfully uploaded dataset to {repo_id} using last resort method")
        
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

