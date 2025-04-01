from datasets import Dataset, Audio, Video
import math # Added for isnan
import os
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
def push_dataset_to_hub(dataset_path, repo_name):
    """
    Push the dataset to the HuggingFace Hub.
    """
    dataset = Dataset.from_file(dataset_path)
    dataset.push_to_hub(repo_name)


# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Push the dataset to the HuggingFace Hub')
    parser.add_argument('--dataset_path', type=str, default='../data/hf_dataset', help='Path to the HuggingFace dataset')
    parser.add_argument('--repo_name', type=str, default='ami-av', help='Name of the repository to push the dataset')
    args = parser.parse_args()
    
    push_dataset_to_hub(args.dataset_path, args.repo_name)

