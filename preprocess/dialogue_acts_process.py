import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import xml.etree.ElementTree as ET
import re
import argparse
from tqdm import tqdm
from collections import defaultdict
from preprocess.constants import DATA_PATH, TRANS_PATH
import pandas as pd

data_path = DATA_PATH # /deepstore/datasets/hmi/speechlaugh-corpus/ami/
transcript_path = TRANS_PATH # /deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts
csv_output_path = os.path.join(DATA_PATH, 'ami_dialogue_acts') # /deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_dialogue_acts

def dialogue_acts_to_csv(
        input_dir=None, # /deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts
        output_dir=None, # /deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_dialogue_acts
        dialogue_acts_dir=None, # /deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts/dialogueActs
        da_types_file=None, # /deepstore/datasets/hmi/speechlaugh-corpus/ami/ontologies/da-types.xml
        include_adjacency_pairs=True, # Whether to process adjacency pairs
        ):
    """
    Process AMI Corpus dialogue acts files and export to CSV.
    
    The CSV will contain: meeting_id, speaker_id, word, start_time, end_time, dialogue_act_type, dialogue_act_gloss, event_type
    And optionally adjacency pairs: source_meeting_speaker, source_dact_id, target_meeting_speaker, target_dact_id, pair_type
    
    Args:
        input_dir: Directory containing the transcript XML files. Default is `data_path/transcripts`
        output_dir: Directory where the CSV files will be saved. Default is `data_path/ami_dialogue_acts`
        dialogue_acts_dir: Directory containing the dialogue acts XML files
        da_types_file: File containing the dialogue act types
        include_adjacency_pairs: Whether to process and include adjacency pairs information
    """
    # Set default directories if not provided
    if not input_dir:
        input_dir = transcript_path
    if not output_dir:
        output_dir = csv_output_path
    if not dialogue_acts_dir:
        dialogue_acts_dir = os.path.join(input_dir, 'dialogueActs')
    if not da_types_file:
        da_types_file = os.path.join(input_dir, 'ontologies/da-types.xml')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dialogue act types from ontology file
    print(f"Loading dialogue act types from {da_types_file}")
    da_types = load_dialogue_act_types(da_types_file)
    
    # Load adjacency pair types if processing adjacency pairs
    ap_types = {}
    if include_adjacency_pairs:
        ap_types_file = os.path.join(input_dir, 'ontologies/ap-types.xml')
        if os.path.exists(ap_types_file):
            print(f"Loading adjacency pair types from {ap_types_file}")
            ap_types = load_adjacency_pair_types(ap_types_file)
        else:
            print(f"Adjacency pair types file not found: {ap_types_file}")
    
    # WORDS FILES
    # Get all words files to determine which meetings and speakers to process
    words_dir = os.path.join(input_dir, 'words')
    words_files = [f for f in os.listdir(words_dir) if f.endswith('.words.xml')]
    print(f"Found {len(words_files)} words files")
    
    # Get all meetings for processing adjacency pairs
    meetings = set()
    meeting_speakers = defaultdict(list)
    for words_file in words_files:
        match = re.match(r'([^.]+)\.([^.]+)\.words\.xml', words_file)
        if match:
            meeting_id, speaker_id = match.groups()
            meetings.add(meeting_id)
            meeting_speakers[meeting_id].append(speaker_id)
    
    # Create CSV files for dialogue acts and adjacency pairs
    da_csv_file = os.path.join(output_dir, 'ami_dialogue_acts.csv')
    
    # Process dialogue acts
    with open(da_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['meeting_id', 'speaker_id', 'dact_id', 'word', 'start_time', 'end_time', 
                           'dialogue_act_type', 'dialogue_act_gloss', 'dialogue_act_category', 'event_type'])
        
        # Process each meeting and speaker
        for words_file in tqdm(words_files, desc="Processing dialogue acts"):
            words_file_path = os.path.join(words_dir, words_file)
            if not os.path.exists(words_file_path):
                print(f"Words file not found: {words_file_path}")
                continue

            # Extract meeting_id and speaker_id from filename
            match = re.match(r'([^.]+)\.([^.]+)\.words\.xml', words_file)
            if not match:
                print(f"Could not parse filename: {words_file}")
                continue
            
            meeting_id, speaker_id = match.groups()

            # Check for corresponding dialogue acts file
            dialogue_acts_file = os.path.join(dialogue_acts_dir, f"{meeting_id}.{speaker_id}.dialog-act.xml")
            
            if not os.path.exists(dialogue_acts_file):
                print(f"Dialogue acts file not found: {dialogue_acts_file}")
                continue
            
            # Process the dialogue acts and write to CSV
            print(f"Processing dialogue acts for {meeting_id} - Speaker {speaker_id}")
            process_dialogue_acts_for_csv(
                csv_writer,
                meeting_id,
                speaker_id,
                words_file_path,
                dialogue_acts_file,
                da_types
            )
    
    print("-"*30)
    print(f"Dialogue acts CSV export completed: {da_csv_file}")
    print("-"*30)
    
    # Process adjacency pairs if requested
    if include_adjacency_pairs:
        ap_csv_file = os.path.join(output_dir, 'ami_adjacency_pairs.csv')
        
        with open(ap_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['meeting_id', 'pair_id', 'pair_type', 'pair_type_gloss',
                               'source_meeting_id', 'source_speaker_id', 'source_dact_id',
                               'target_meeting_id', 'target_speaker_id', 'target_dact_id'])
            
            # Process each meeting's adjacency pairs
            for meeting_id in tqdm(meetings, desc="Processing adjacency pairs"):
                adjacency_file = os.path.join(dialogue_acts_dir, f"{meeting_id}.adjacency-pairs.xml")
                
                if os.path.exists(adjacency_file):
                    print(f"Processing adjacency pairs for {meeting_id}")
                    process_adjacency_pairs_for_csv(
                        csv_writer,
                        meeting_id,
                        adjacency_file,
                        ap_types
                    )
                else:
                    print(f"Adjacency pairs file not found: {adjacency_file}")
        
        print("-"*30)
        print(f"Adjacency pairs CSV export completed: {ap_csv_file}")
        print("-"*30)

    print("-"*30)
    print("Join dialogue acts and adjacency pairs CSV files")
    print("-"*30)

    # Join dialogue acts and adjacency pairs CSV files
    da_csv_file = os.path.join(output_dir, 'ami_dialogue_acts.csv')
    ap_csv_file = os.path.join(output_dir, 'ami_adjacency_pairs.csv')

    da_df = pd.read_csv(da_csv_file)
    ap_df = pd.read_csv(ap_csv_file)
    ap_df.rename(columns={'source_dact_id': 'dact_id', 'source_speaker_id': 'speaker_id'}, inplace=True)
    ap_df.drop(columns=['source_meeting_id', 'target_meeting_id'], inplace=True)

    # Join on meeting_id
    joined_df = pd.merge(da_df, ap_df, on=['meeting_id', 'speaker_id', 'dact_id'], how='left')

    # Save joined CSV file
    joined_csv_file = os.path.join(output_dir, 'ami_da_ap_laughter.csv')
    joined_df.to_csv(joined_csv_file, index=False)

    print("-"*30)
    print(f"Joined CSV file saved: {joined_csv_file}")
    print("-"*30)


def process_dialogue_acts_for_csv(
        csv_writer,
        meeting_id, 
        speaker_id, 
        words_file,
        dialogue_acts_file,
        da_types
        ):
    """
    Process dialogue acts files and write to CSV.
    
    Args:
        csv_writer: CSV writer object
        meeting_id: Meeting ID (e.g. ES2002a)
        speaker_id: Speaker ID (e.g. A)
        words_file: Path to the words XML file
        dialogue_acts_file: Path to the dialogue acts XML file
        da_types: Dictionary mapping dialogue act type IDs to information
    """
    # Parse XML files
    words_tree = ET.parse(words_file)
    words_root = words_tree.getroot()
    
    da_tree = ET.parse(dialogue_acts_file)
    da_root = da_tree.getroot()
    
    # Define namespace mapping
    ns = {'nite': 'http://nite.sourceforge.net/'}

    punctuations = ['.', '?', '!', ':', ';', ',', '(', ')', '[', ']', '{', '}', '~', '`']
  
    # Create word mapping for quick lookup
    word_elements = {}
    for word_elem in words_root.findall('.//*[@{{{0}}}id]'.format(ns['nite']), ns):
        word_id = word_elem.get('{{{0}}}id'.format(ns['nite']))
        if word_elem.tag in ['w', 'vocalsound']:
            word_elements[word_id] = word_elem
    
    # Process each dialogue act
    for dact_elem in da_root.findall('.//dact', ns):
        dact_id = dact_elem.get('{{{0}}}id'.format(ns['nite']))
        
        # Get dialogue act type
        da_type_elem = dact_elem.find('.//nite:pointer[@role="da-aspect"]', ns)
        da_type_id = None
        da_type_info = {'name': '', 'gloss': '', 'category': ''}
        
        if da_type_elem is not None:
            da_type_ref = da_type_elem.get('href')
            da_type_match = re.search(r'#id\(([^)]+)\)', da_type_ref)
            if da_type_match:
                da_type_id = da_type_match.group(1)
                da_type_info = da_types.get(da_type_id, {'name': '', 'gloss': '', 'category': ''})
        
        # Get affected words
        child_elems = dact_elem.findall('.//nite:child', ns)
        for child_elem in child_elems:
            href = child_elem.get('href')
            if href:
                # Extract word IDs from href
                word_ids = extract_word_ids_from_href(href)
                
                # Process each word
                for word_id in word_ids:
                    if word_id in word_elements:
                        word_elem = word_elements[word_id]
                        
                        # Initialize event type as fluent by default
                        event_type = 'fluent'
                        
                        # Get word text and determine event type
                        if word_elem.tag == 'w':
                            text = word_elem.text if word_elem.text else ""

                            if text in punctuations:
                                continue
                            
                            text = ''.join(text.split('_')) # remove underscores, for example: T_V -> TV
                            text = text.replace('&#39;', "'")  # Replace HTML entities
                            text = text.replace('&quot;', '"')  # Replace HTML entities
  
                            event_type = 'fluent'  # Regular words are fluent
                        elif word_elem.tag == 'vocalsound':
                            if word_elem.get('type') == 'laugh':
                                text = "<laugh>"
                                event_type = 'laughter'  # Laughter events
                            else:
                                text = f"<{word_elem.get('type', 'vocalsound')}>"
                                event_type = 'vocalsound'  # Other vocal sounds
                        else:
                            continue
                        
                        # Get timing information
                        start_time = word_elem.get('starttime', '')
                        end_time = word_elem.get('endtime', '')
                        
                        # Write to CSV
                        csv_writer.writerow([
                            meeting_id,
                            speaker_id,
                            dact_id,
                            text,
                            start_time,
                            end_time,
                            da_type_info['name'],
                            da_type_info['gloss'],
                            da_type_info['category'],
                            event_type
                        ])


def process_adjacency_pairs_for_csv(
        csv_writer,
        meeting_id,
        adjacency_file,
        ap_types
        ):
    """
    Process adjacency pairs file and write to CSV.
    
    Args:
        csv_writer: CSV writer object
        meeting_id: Meeting ID
        adjacency_file: Path to the adjacency pairs XML file
        ap_types: Dictionary mapping adjacency pair type IDs to names
    """
    try:
        ap_tree = ET.parse(adjacency_file)
        ap_root = ap_tree.getroot()
        
        # Define namespace mapping
        ns = {'nite': 'http://nite.sourceforge.net/'}
        
        # Process each adjacency pair
        for pair_elem in ap_root.findall('.//adjacency-pair', ns):
            pair_id = pair_elem.get('{{{0}}}id'.format(ns['nite']))
            
            # Get pair type
            type_elem = pair_elem.find('.//nite:pointer[@role="type"]', ns)
            pair_type_id = ''
            pair_type_gloss = ''
            
            if type_elem is not None:
                type_ref = type_elem.get('href')
                type_match = re.search(r'#id\(([^)]+)\)', type_ref)
                if type_match:
                    pair_type_id = type_match.group(1)
                    pair_type_gloss = ap_types.get(pair_type_id, '')
            
            # Get source and target
            source_elem = pair_elem.find('.//nite:pointer[@role="source"]', ns)
            target_elem = pair_elem.find('.//nite:pointer[@role="target"]', ns)
            
            source_info = extract_dact_info_from_href(source_elem.get('href') if source_elem is not None else '')
            target_info = extract_dact_info_from_href(target_elem.get('href') if target_elem is not None else '')
            
            # Write to CSV
            csv_writer.writerow([
                meeting_id,
                pair_id,
                pair_type_id,
                pair_type_gloss,
                source_info['meeting_id'],
                source_info['speaker_id'],
                source_info['dact_id'],
                target_info['meeting_id'],
                target_info['speaker_id'],
                target_info['dact_id']
            ])
            
    except Exception as e:
        print(f"Error processing adjacency pairs file {adjacency_file}: {e}")


def load_dialogue_act_types(da_types_file):
    """
    Load dialogue act types from the ontology file.
    
    Returns a dictionary mapping dialogue act IDs to their information.
    """
    da_types = {}
    
    try:
        tree = ET.parse(da_types_file)
        root = tree.getroot()
        
        # Define namespace mapping
        ns = {'nite': 'http://nite.sourceforge.net/'}
        
        # Extract categories and their dialogue acts
        for category_elem in root.findall('./da-type', ns):
            # category_id = category_elem.get('{{{0}}}id'.format(ns['nite']))
            # category_name = category_elem.get('name', '')
            category_gloss = category_elem.get('gloss', '')
            
            # Process subcategory dialogue acts
            for da_elem in category_elem.findall('./da-type', ns):
                da_id = da_elem.get('{{{0}}}id'.format(ns['nite']))
                da_name = da_elem.get('name', '')
                da_gloss = da_elem.get('gloss', '')
                
                if da_id:
                    da_types[da_id] = {
                        'name': da_name,
                        'gloss': da_gloss,
                        'category': category_gloss
                    }
                    
    except Exception as e:
        print(f"Error loading dialogue act types: {e}")
    
    print(f"Loaded {len(da_types)} dialogue act types")
    return da_types


def load_adjacency_pair_types(ap_types_file):
    """
    Load adjacency pair types from the ontology file.
    
    Returns a dictionary mapping adjacency pair type IDs to names.
    """
    ap_types = {}
    
    try:
        tree = ET.parse(ap_types_file)
        root = tree.getroot()
        
        # Define namespace mapping
        ns = {'nite': 'http://nite.sourceforge.net/'}
        
        # Extract each adjacency pair type
        for ap_elem in root.findall('./ap-type', ns):
            ap_id = ap_elem.get('{{{0}}}id'.format(ns['nite']))
            ap_gloss = ap_elem.get('gloss', '')
            if ap_id and ap_gloss:
                ap_types[ap_id] = ap_gloss
                
    except Exception as e:
        print(f"Error loading adjacency pair types: {e}")
    
    print(f"Loaded {len(ap_types)} adjacency pair types")
    return ap_types


def extract_word_ids_from_href(href):
    """
    Extract word IDs from href reference.
    
    Args:
        href: Reference string (e.g., "ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)")
    
    Returns:
        List of word IDs
    """
    word_ids = []
    
    # Handle single word reference
    single_match = re.search(r'#id\(([^)]+)\)$', href)
    if single_match:
        word_ids.append(single_match.group(1))
        return word_ids
    
    # Handle word range reference
    range_match = re.search(r'#id\(([^)]+)\)\.\.id\(([^)]+)\)', href)
    if range_match:
        start_id, end_id = range_match.groups()
        
        # Extract numeric part of IDs to determine range
        start_match = re.search(r'words(\d+)', start_id)
        end_match = re.search(r'words(\d+)', end_id)
        
        if start_match and end_match:
            start_num = int(start_match.group(1))
            end_num = int(end_match.group(1))
            
            # Generate all word IDs in the range
            prefix = start_id.split('words')[0]
            for i in range(start_num, end_num + 1):
                word_id = f"{prefix}words{i}"
                word_ids.append(word_id)
    
    return word_ids


def extract_dact_info_from_href(href):
    """
    Extract dialogue act information from href reference.
    
    Args:
        href: Reference string (e.g., "ES2002a.A.dialog-act.xml#id(ES2002a.A.dialog-act.dharshi.1)")
    
    Returns:
        Dictionary with meeting_id, speaker_id, and dact_id
    """
    info = {'meeting_id': '', 'speaker_id': '', 'dact_id': ''}
    
    if not href:
        return info
    
    # Extract dact ID
    dact_match = re.search(r'#id\(([^)]+)\)', href)
    if dact_match:
        info['dact_id'] = dact_match.group(1)
    
    # Extract meeting and speaker from filename
    file_match = re.search(r'([^.]+)\.([^.]+)\.dialog-act\.xml', href)
    if file_match:
        info['meeting_id'] = file_match.group(1)
        info['speaker_id'] = file_match.group(2)
    
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AMI Corpus dialogue acts data to CSV")
    parser.add_argument("--input", default=transcript_path, help="Input directory containing transcript XML files")
    parser.add_argument("--output", default=csv_output_path, help="Output directory for CSV files")
    parser.add_argument("--dialogue_acts_dir", help="Directory containing dialogue acts XML files")
    parser.add_argument("--da_types_file", help="File containing dialogue act types")
    parser.add_argument("--include_adjacency_pairs", action='store_true', default=True, help="Include adjacency pairs processing")
    parser.add_argument("--skip_adjacency_pairs", action='store_true', help="Skip adjacency pairs processing")
    
    args = parser.parse_args()
    
    # Handle adjacency pairs flag
    include_adjacency_pairs = args.include_adjacency_pairs and not args.skip_adjacency_pairs
    
    dialogue_acts_to_csv(
        input_dir=args.input,
        output_dir=args.output,
        dialogue_acts_dir=args.dialogue_acts_dir,
        da_types_file=args.da_types_file,
        include_adjacency_pairs=include_adjacency_pairs
    ) 