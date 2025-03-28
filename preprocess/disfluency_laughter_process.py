import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import xml.etree.ElementTree as ET
import re
import argparse

from preprocess.constants import DATA_PATH, TRANS_PATH

data_path = DATA_PATH
transcript_path = TRANS_PATH
csv_output_path = os.path.join(DATA_PATH, 'dsfl_laugh')

def disfluency_laughter_to_csv(input_dir=None, output_dir=None):
    """
    Process AMI Corpus transcript files and export the disfluency and laughter categories to CSV.
    
    The CSV will contain: meeting_id, speaker_id, word, start_time, end_time, disfluency_type, is_laugh
    
    Args:
        input_dir: Directory containing the transcript XML files. Default is `data_path/transcripts`
        output_dir: Directory where the CSV files will be saved. Default is `data_path/dsfl_laugh`
    """
    # Set default directories if not provided
    if not input_dir:
        input_dir = transcript_path
    if not output_dir:
        output_dir = csv_output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load disfluency types from ontology file
    dsfl_types = load_disfluency_types(os.path.join(input_dir, 'ontologies/dsfl-types.xml')) # a dictionary of disfluency type name
    
    # Get all words files to determine which meetings and speakers to process
    words_dir = os.path.join(input_dir, 'words')
    words_files = [f for f in os.listdir(words_dir) if f.endswith('.words.xml')]
    print(f"Found {len(words_files)} words files")
    
    # Create a CSV file for all processed data
    csv_file_path = os.path.join(output_dir, 'disfluency_laughter_markers.csv')
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        # Define CSV writer and header
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['meeting_id', 'speaker_id', 'word', 'start_time', 'end_time', 'disfluency_type', 'is_laugh'])
        
        # Process each meeting and speaker
        for words_file in words_files:
            # Extract meeting_id and speaker_id from filename
            words_file_path = os.path.join(words_dir, words_file)
            if not os.path.exists(words_file_path):
                print(f"Words file not found: {words_file_path}")
                continue

            # Example: ES2002a.A.words.xml -> meeting_id=ES2002a, speaker_id=A
            match = re.match(r'([^.]+)\.([^.]+)\.words\.xml', words_file)
            if not match:
                print(f"Could not parse filename: {words_file}")
                continue
            
            meeting_id, speaker_id = match.groups()
            # Check for corresponding disfluency file
            disfluency_file = os.path.join(input_dir, 'disfluency', f"{meeting_id}.{speaker_id}.disfluency.xml")
            
            
            # Process the transcript and write to CSV
            print(f"Processing {meeting_id} - Speaker {speaker_id}")
            process_disfluency_laughter_for_csv(
                meeting_id,
                speaker_id,
                words_file_path,  # Pass the full path to the words file
                disfluency_file if os.path.exists(disfluency_file) else None,
                dsfl_types,
                csv_writer
            )
    
    print(f"CSV export completed: {csv_file_path}")


def process_disfluency_laughter_for_csv(meeting_id, speaker_id, words_file, disfluency_file, dsfl_types, csv_writer):
    """
    Process transcript files and write disfluency and laughter categories to CSV.
    
    Args:
        meeting_id: Meeting ID (e.g. ES2002a)
        speaker_id: Speaker ID (e.g. A)
        words_file: Path to the words XML file (e.g. ES2002a.A.words.xml)
        disfluency_file: Path to the disfluency XML file (e.g. ES2002a.A.disfluency.xml) - can be None
        dsfl_types: Dictionary mapping disfluency type IDs to names (e.g. ami_dsfl_1 -> "delete")
        csv_writer: CSV writer object to write data
    """
    # Parse XML files
    words_tree = ET.parse(words_file)
    words_root = words_tree.getroot()
    
    # Define namespace mapping
    ns = {'nite': 'http://nite.sourceforge.net/'}
    
    # Create disfluency mapping if disfluency file exists
    word_to_disfluency = {}
    if disfluency_file:
        try:
            disfluency_tree = ET.parse(disfluency_file)
            disfluency_root = disfluency_tree.getroot()
            
            # Process each disfluency annotation
            for dsfl_elem in disfluency_root.findall('.//dsfl', ns):
                # Get disfluency type
                dsfl_type_elem = dsfl_elem.find('.//nite:pointer[@role="dsfl-type"]', ns)
                if dsfl_type_elem is not None:
                    dsfl_type_ref = dsfl_type_elem.get('href')
                    dsfl_type_id = re.search(r'#id\(([^)]+)\)', dsfl_type_ref)
                    if dsfl_type_id:
                        dsfl_type_id = dsfl_type_id.group(1)
                    
                    # Get affected words
                    for child_elem in dsfl_elem.findall('.//nite:child', ns):
                        href = child_elem.get('href')
                        if href:
                            # Handle single word reference
                            single_match = re.search(r'#id\(([^)]+)\)', href)
                            if single_match:
                                word_id = single_match.group(1)
                                word_to_disfluency[word_id] = dsfl_type_id
                            
                            # Handle word range reference
                            range_match = re.search(r'#id\(([^)]+)\)\.\.id\(([^)]+)\)', href)
                            if range_match:
                                start_id, end_id = range_match.groups()
                                
                                # Extract numeric part of IDs to determine range
                                start_num = int(re.search(r'words(\d+)', start_id).group(1))
                                end_num = int(re.search(r'words(\d+)', end_id).group(1))
                                
                                # Mark all words in the range
                                prefix = start_id.split('words')[0]
                                for i in range(start_num, end_num + 1):
                                    word_id = f"{prefix}words{i}"
                                    word_to_disfluency[word_id] = dsfl_type_id
        except Exception as e:
            print(f"Error processing disfluency file: {e}")
    
    # Process each word and append to CSV
    for word_elem in words_root.findall('.//*[@{{{0}}}id]'.format(ns['nite']), ns):
        word_id = word_elem.get('{{{0}}}id'.format(ns['nite']))
        
        # Only process word elements and vocal sounds
        if word_elem.tag not in ['w', 'vocalsound']:
            continue
        
        # Get word text
        if word_elem.tag == 'w':
            text = word_elem.text if word_elem.text else ""
            text = text.replace('&#39;', "'")  # Replace HTML entities
            is_laugh = False
        elif word_elem.tag == 'vocalsound' and word_elem.get('type') == 'laugh':
            text = "<laugh>"
            is_laugh = True
        else:
            continue  # Skip other vocal sounds
        
        # Get timing information
        start_time = word_elem.get('starttime', '')
        end_time = word_elem.get('endtime', '')
        
        # Get disfluency type
        disfluency_id = word_to_disfluency.get(word_id, '')
        disfluency_type = dsfl_types.get(disfluency_id, '')
        
        # Only write to CSV if the word has a disfluency type or is a laugh
        if disfluency_type or is_laugh:
            # Write to CSV
            csv_writer.writerow([
                meeting_id,
                speaker_id,
                text,
                start_time,
                end_time,
                disfluency_type, # 19 disfluency types
                1 if is_laugh else 0 # 1 if laughter, 0 otherwise
            ])

def load_disfluency_types(dsfl_types_file):
    """
    Load disfluency types from the ontology file.
    
    Returns a dictionary of 19 disfluency types (annotated by the AMI Corpus)
    """
    dsfl_types = {}
    
    try:
        tree = ET.parse(dsfl_types_file)
        root = tree.getroot()
        
        # Define namespace mapping
        ns = {'nite': 'http://nite.sourceforge.net/'}
        
        # Extract each disfluency type
        for dsfl_elem in root.findall('./dsfl-type', ns):
            dsfl_id = dsfl_elem.get('{{{0}}}id'.format(ns['nite']))
            dsfl_name = dsfl_elem.get('name')
            if dsfl_id and dsfl_name:
                dsfl_types[dsfl_id] = dsfl_name
    except Exception as e:
        print(f"Error loading disfluency types: {e}")
    
    return dsfl_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AMI Corpus transcript data to CSV")
    parser.add_argument("--input", default=transcript_path, help="Input directory containing transcript XML files")
    parser.add_argument("--output", default=csv_output_path, help="Output directory for CSV files")
    
    args = parser.parse_args()
    disfluency_laughter_to_csv(args.input, args.output) 