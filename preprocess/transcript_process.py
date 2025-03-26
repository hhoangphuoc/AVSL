import os
import xml.etree.ElementTree as ET
import re
import argparse
from tqdm import tqdm
from constants import DATA_PATH, TRANS_PATH, TRANS_SEG_PATH

data_path = DATA_PATH
transcript_path = TRANS_PATH
transcript_segments_dir = TRANS_SEG_PATH


def process_transcripts(input_dir=None, output_dir=None):
    """
    Process all original transcript files and store the sentence-level transcripts 
    based on timestamps, each output file refer to a single speaker's transcript (named as `meeting_id-speaker_id.txt`)

    For example, the output file `EN2001a-A.txt` stores all sentences spoken by speaker A in meeting EN2001a.
    
    Args:
        input_dir: Directory containing the transcript XML files. Default is `data_path/transcripts`
        output_dir: Directory where the output text files will be saved. Default is `data_path/transcript_segments`
    """
    # Set default directories if not provided
    if not input_dir:
        input_dir = transcript_path
    if not output_dir:
        output_dir = transcript_segments_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all segment files
    segments_dir = os.path.join(input_dir, 'segments')
    segments_files = [f for f in os.listdir(segments_dir) if f.endswith('.segments.xml')]
    
    # Process each segment file
    for segment_file in segments_files:
        # Extract meeting_id and speaker_id from filename
        # Example: EN2001a.A.segments.xml -> meeting_id=EN2001a, speaker_id=A
        match = re.match(r'([^.]+)\.([^.]+)\.segments\.xml', segment_file)
        if not match:
            print(f"Could not parse filename: {segment_file}")
            continue
        
        meeting_id, speaker_id = match.groups()
        
        # Path to corresponding words file
        words_file = os.path.join(input_dir, 'words', f"{meeting_id}.{speaker_id}.words.xml")
        
        if not os.path.exists(words_file):
            print(f"Words file not found: {words_file}")
            continue
            
        # Output file path
        output_file = os.path.join(output_dir, f"{meeting_id}-{speaker_id}.txt")
        
        print(f"Processing {meeting_id} - Speaker {speaker_id}")
        process_transcript_files(
            os.path.join(segments_dir, segment_file),
            words_file,
            output_file
        )
        print(f"Output written to {output_file}")


def process_transcript_files(segment_file, words_file, output_file):
    """
    Process a pair of segment and words XML files to create a text transcript.
    
    Args:
        segment_file: Path to the segments XML file
        words_file: Path to the words XML file
        output_file: Path to the output text file
    """
    # Parse XML files
    segments_tree = ET.parse(segment_file)
    segments_root = segments_tree.getroot()
    
    words_tree = ET.parse(words_file)
    words_root = words_tree.getroot()
    
    # Define namespace mapping
    ns = {'nite': 'http://nite.sourceforge.net/'}
    
    # Create a dictionary of all words by ID for quick lookup
    word_dict = {}
    for word_elem in words_root.findall('.//w', ns):
        word_id = word_elem.get('{{{0}}}id'.format(ns['nite']))
        text = word_elem.text if word_elem.text else ""
        # Replace HTML entities
        text = text.replace('&#39;', "'")
        word_dict[word_id] = text
    
    # Add vocalsound elements with type="laugh" to the word dictionary
    # tag: <vocalsound ... type="laugh" />
    for vocal_elem in words_root.findall('.//vocalsound[@type="laugh"]', ns):
        vocal_id = vocal_elem.get('{{{0}}}id'.format(ns['nite']))
        # Add a laugh marker to the word dictionary
        word_dict[vocal_id] = "<laugh>"
        
    # Create dictionaries for punctuation, truncationand disfluency markers

    # Handling word that is punctuation: 
    # tag: <w ... punc="true" /> 
    punct_dict = {}
    for word_elem in words_root.findall('.//w[@punc="true"]', ns):
        word_id = word_elem.get('{{{0}}}id'.format(ns['nite']))
        punct_dict[word_id] = True
    
    # Handling word that is truncate: 
    # tag: <w ... trunc="true" />
    # Note that the truncate word will be removed from the transcript   
    trunc_dict = {}
    for word_elem in words_root.findall('.//w[@trunc="true"]', ns):
        word_id = word_elem.get('{{{0}}}id'.format(ns['nite']))
        trunc_dict[word_id] = True
    
    # Handling word that is disfluency marker
    # tag: <disfmarker ... />
    disfluency_dict = {}
    for disf_elem in words_root.findall('.//disfmarker', ns):
        disf_id = disf_elem.get('{{{0}}}id'.format(ns['nite'])) 
        disfluency_dict[disf_id] = True
    
    # Get all element IDs in order of appearance
    all_elements = {}
    for elem in words_root.findall('.//*[@{{{0}}}id]'.format(ns['nite']), ns):
        elem_id = elem.get('{{{0}}}id'.format(ns['nite']))
        if elem_id and elem.tag in ['w', 'vocalsound', 'disfmarker']:
            # Extract numeric part to determine ordering
            match = re.search(r'words(\d+)', elem_id)
            if match:
                elem_num = int(match.group(1))
                all_elements[elem_id] = elem_num
    
    # Process segments and write to output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for segment in segments_root.findall('.//segment', ns):
            # transcribe start and end time of the segment
            start_time = segment.get('transcriber_start')
            end_time = segment.get('transcriber_end')
            
            # Get child element that contains references to word IDs
            child_elem = segment.find('.//nite:child', ns)
            if child_elem is None:
                continue
                
            # Extract word ID references, format is like: 
            # "EN2001a.A.words.xml#id(EN2001a.A.words0)..id(EN2001a.A.words1)"
            href = child_elem.get('href')
            if not href:
                continue
                
            # Extract word range
            match = re.search(r'#id\(([^)]+)\)\.\.id\(([^)]+)\)', href) #pattern: #id(EN2001a.A.words0)..id(EN2001a.A.words3)
            if not match:
                continue
                
            start_id, end_id = match.groups() #start_id = EN2001a.A.words0, end_id = EN2001a.A.words3
            
            # Extract numeric part of IDs to determine range
            start_num = int(re.search(r'words(\d+)', start_id).group(1)) #start_num = 0
            end_num = int(re.search(r'words(\d+)', end_id).group(1)) #end_num = 3
            
            # Collect all elements in this segment's range
            segment_elements = []
            prefix = start_id.split('words')[0] #prefix = EN2001a.A. #this using to extract the prefix id, and joining it with the word id
            
            for i in range(start_num, end_num + 1):
                elem_id = f"{prefix}words{i}" #elem_id = EN2001a.A.words0 with start_num = 0
                
                # Skip disfluency markers
                if elem_id in disfluency_dict:
                    continue
                
                # Skip truncate words   
                if elem_id in trunc_dict:
                    continue
                
                # Check if this ID represents a word or a laugh
                if elem_id in word_dict:
                    word = word_dict[elem_id]
                    is_punct = elem_id in punct_dict
                    segment_elements.append((elem_id, word, is_punct))
            
            # Sort elements by their ID number to maintain correct order
            segment_elements.sort(key=lambda x: all_elements.get(x[0], 0))
            
            # Reconstruct the segment text
            segment_text = []
            
            for elem_id, word, is_punct in segment_elements:
                # Don't add space before punctuation
                if is_punct and segment_text:
                    segment_text[-1] = segment_text[-1] + word
                else:
                    segment_text.append(word)
            
            # Join words with spaces and write to file
            if segment_text:
                # Add timing information as metadata
                out_file.write(f"[{start_time}-{end_time}] {' '.join(segment_text)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AMI Corpus transcripts")
    parser.add_argument("--input", default=transcript_path, help="Input directory containing transcript XML files")
    parser.add_argument("--output", default=transcript_segments_dir, help="Output directory for text transcripts")
    
    args = parser.parse_args()
    process_transcripts(args.input, args.output)





