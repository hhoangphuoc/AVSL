#!/usr/bin/env python3
"""
Filter <laugh> events from AMI dataset, find media paths, and generate statistics.

This script filters out entries from a given CSV file that correspond to laughter events.
It generates a 'segment_id' for each event and then searches for corresponding
audio, video, and lip-video files on the filesystem.

The script produces a clean CSV file containing only complete rows with all media paths
and other relevant metadata. It also generates detailed statistics on the filtered data.
"""

import pandas as pd
import argparse
import sys
import os
import subprocess
from tqdm import tqdm

# Configure pandas to use tqdm for progress_apply
tqdm.pandas()

def cache_media_paths():
    """
    Find all audio, video, and lip video paths using the 'find' command and cache them.
    This is much more efficient than running 'find' for each segment individually.
    """
    print("Caching all media file paths. This may take a moment...")
    
    # Base path for media segments
    base_path = "/deepstore/datasets/hmi/speechlaugh-corpus/ami/fluent_laughter/segments/"
    
    audio_paths = {}
    video_paths = {}
    lip_video_paths = {}

    # A single 'find' command for all relevant files is most efficient
    cmd = f"find {base_path}/chunk_* -type f \\( -name '*.wav' -o -name '*.mp4' \\)"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        all_files = result.stdout.strip().split('\n')
        
        print(f"Found {len(all_files)} potential media files. Processing...")

        for path in tqdm(all_files, desc="Categorizing media files"):
            if not path:
                continue
            
            dir_name = os.path.dirname(path)
            fname_ext = os.path.basename(path)
            fname, ext = os.path.splitext(fname_ext)
            
            if ext == '.wav' and 'audio_segments' in dir_name and 'laughter' in fname:
                audio_paths[fname] = path
            elif ext == '.mp4' and 'video_segments' in dir_name and 'laughter' in fname:
                video_paths[fname] = path
            elif ext == '.mp4' and 'lip_segments' in dir_name and '-laughter_lip' in fname:
                segment_id = fname.replace('_lip', '')
                lip_video_paths[segment_id] = path

    except subprocess.CalledProcessError as e:
        print(f"Error executing find command to cache media paths: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        print("Continuing with empty media paths. Output will likely be empty.", file=sys.stderr)
        return {}, {}, {}

    print(f"Cached paths: {len(audio_paths)} audio, {len(video_paths)} video, {len(lip_video_paths)} lip video.")
    return audio_paths, video_paths, lip_video_paths

def filter_laugh_events(input_file, output_file):
    """
    Filter laugh events, find corresponding media files, and save to a clean CSV.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to output CSV file for filtered events.
    
    Returns:
        pd.DataFrame: DataFrame containing filtered laugh events with media paths.
    """
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Total rows in dataset: {len(df)}")
    
    # Filter for laugh events
    laugh_filter = (df['word'] == '<laugh>') & (df['event_type'] == 'laughter')
    laugh_events = df[laugh_filter].copy()
    
    print(f"Total <laugh> events found: {len(laugh_events)}")
    
    if laugh_events.empty:
        print("No laugh events found to process.")
        return pd.DataFrame()

    # Generate segment_id
    print("Generating segment_id for each laugh event...")
    laugh_events['segment_id'] = laugh_events.progress_apply(
        lambda row: f"{row['meeting_id']}-{row['speaker_id']}-{row['start_time']:.2f}-{row['end_time']:.2f}-laughter",
        axis=1
    )
    
    # Get all media paths efficiently
    audio_paths, video_paths, lip_video_paths = cache_media_paths()

    # Add media path columns by mapping from segment_id
    print("Mapping media paths to laugh events...")
    laugh_events['audio'] = laugh_events['segment_id'].map(audio_paths)
    laugh_events['video'] = laugh_events['segment_id'].map(video_paths)
    laugh_events['lip_video'] = laugh_events['segment_id'].map(lip_video_paths)

    # Define final columns for the output CSV ----------------------------------------------
    # COLUMNS TO KEEP (TASK 1 - DA)
    # final_columns = [
    #     'segment_id', 'meeting_id', 'speaker_id', 'dact_id', 'word', 'start_time', 'end_time',
    #     'dialogue_act_type', 'dialogue_act_gloss', 'dialogue_act_category', 'event_type',
    #     'audio', 'video', 'lip_video'
    # ]
    # --------------------------------------------------------------------------------------
    # COLUMNS TO KEEP (TASK 2 - DA + AP) --------------------------------------------------
    final_columns = [
        'segment_id', 'meeting_id', 'speaker_id', 'dact_id', 'word', 'start_time', 'end_time',
        'dialogue_act_type', 'dialogue_act_gloss', 'dialogue_act_category', 'event_type',
        'pair_id', 'pair_type', 'pair_type_gloss', 'target_speaker_id', 'target_dact_id',
        'audio', 'video', 'lip_video'
    ]
    # --------------------------------------------------------------------------------------

    # Use only the columns that exist in the input dataframe, plus the new ones
    existing_columns = [col for col in final_columns if col in laugh_events.columns]
    
    processed_df = laugh_events[existing_columns]
    print(f"Created dataframe with columns: {', '.join(existing_columns)}")



    # TODO: Uncomment this when we have a way to handle missing values----------------------
    # Drop rows that have any missing values in any column
    rows_before_drop = len(processed_df)


    # Filter out rows that missing the following columns:
    # audio, video, lip_video
    processed_df = processed_df.dropna(subset=['audio', 'video', 'lip_video'])

    rows_after_drop = len(processed_df)
    
    print(f"Dropped {rows_before_drop - rows_after_drop} rows with missing data in selected columns.")
    print(f"Final dataset has {rows_after_drop} complete rows.")
    # --------------------------------------------------------------------------------------

    # Save final dataframe to CSV
    processed_df.to_csv(output_file, index=False)
    print(f"Filtered laugh events with all media paths saved to {output_file}")
    
    return processed_df

def generate_statistics(laugh_events):
    """
    Generate statistics about various aspects of the laugh events.
    """
    print("\n" + "="*60)
    print("LAUGH EVENT STATISTICS")
    print("="*60)
    
    available_columns = laugh_events.columns.tolist()
    stats = {}
    print(f"Available columns for statistics: {', '.join(available_columns)}")

    # Dialogue Act Statistics
    if 'dialogue_act_type' in available_columns:
        stats['dialogue_act_type'] = laugh_events['dialogue_act_type'].value_counts()
    if 'dialogue_act_gloss' in available_columns:
        stats['dialogue_act_gloss'] = laugh_events['dialogue_act_gloss'].value_counts()
    
    # Adjacency Pair Statistics
    if 'pair_id' in available_columns:
        events_with_pairs = laugh_events.dropna(subset=['pair_id'])
        stats['events_with_pairs_count'] = len(events_with_pairs)
        if not events_with_pairs.empty:
            if 'pair_type' in available_columns:
                stats['pair_type'] = events_with_pairs['pair_type'].value_counts()
            if 'pair_type_gloss' in available_columns:
                stats['pair_type_gloss'] = events_with_pairs['pair_type_gloss'].value_counts()
    
    # Speaker and Meeting Statistics
    if 'speaker_id' in available_columns:
        stats['speaker'] = laugh_events['speaker_id'].value_counts()
    if 'meeting_id' in available_columns:
        stats['meeting'] = laugh_events['meeting_id'].value_counts()
        
    # Media Path Statistics
    media_stats = {}
    for media_col in ['audio', 'video', 'lip_video']:
        if media_col in available_columns:
            media_stats[media_col] = laugh_events[media_col].notna().sum()
    stats['media_paths'] = media_stats
    
    return stats

def save_statistics_to_files(laugh_events, stats, output_prefix):
    """
    Save detailed statistics to separate CSV files.
    """
    print(f"\nSaving statistics with prefix '{output_prefix}'...")

    # Ensure output directory exists
    stats_dir = os.path.dirname(output_prefix)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    
    total_events = len(laugh_events)
    if total_events == 0:
        print("No data to generate statistics for.")
        return
    
    for key, data in stats.items():
        if isinstance(data, pd.Series) and not data.empty:
            df = data.reset_index()
            # Special handling for meeting and speaker stats column names
            if key in ['meeting', 'speaker']:
                col_name = f"{key}_id"
            else:
                col_name = key
            df.columns = [col_name, 'count']
            
            # Calculate percentage
            if 'pair' in key:
                pair_count = stats.get('events_with_pairs_count', total_events)
                df['percentage'] = (df['count'] / pair_count) * 100 if pair_count > 0 else 0
            else:
                df['percentage'] = (df['count'] / total_events) * 100
            
            output_path = f"{output_prefix}_{key}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {key} statistics to {output_path}")

    if 'media_paths' in stats and stats['media_paths']:
        media_df = pd.DataFrame.from_dict(stats['media_paths'], orient='index', columns=['count'])
        media_df['percentage'] = (media_df['count'] / total_events) * 100
        output_path = f"{output_prefix}_media_paths.csv"
        media_df.to_csv(output_path)
        print(f"Saved media path statistics to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Filter <laugh> events, find all media paths, and generate statistics.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_file', help='Input CSV file path (e.g., from AMI dataset annotations).')
    parser.add_argument(
        '-o', '--output', default='laugh_events_filtered.csv', 
        help='Output CSV file for filtered events.\n(default: laugh_events_filtered.csv)'
    )
    parser.add_argument(
        '-s', '--stats-prefix', default='laugh_stats',
        help='Prefix for statistics output files.\n(default: laugh_stats)'
    )
    parser.add_argument(
        '--no-stats-files', action='store_true',
        help='If set, do not save statistics to separate files.'
    )
    
    args = parser.parse_args()
    
    try:
        laugh_events = filter_laugh_events(args.input_file, args.output)
        
        if not laugh_events.empty:
            stats = generate_statistics(laugh_events)
            
            if not args.no_stats_files:
                save_statistics_to_files(laugh_events, stats, args.stats_prefix)
            
            print("\n" + "="*60)
            print("PROCESSING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Filtered events saved to: {args.output}")
            if not args.no_stats_files:
                print(f"Statistics files created with prefix: {args.stats_prefix}")
        else:
            print("\nProcessing finished, but no complete laugh events were found.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 