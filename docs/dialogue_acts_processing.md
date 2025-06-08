# Dialogue Acts Processing Documentation

This document explains how to use the dialogue acts processing function to extract and analyze dialogue acts and adjacency pairs from the AMI Meeting Corpus.

## Overview

The dialogue acts processing pipeline extracts structured information from AMI corpus XML annotations, including:
- **Dialogue Acts**: Categorized speech acts (e.g., Inform, Suggest, Assess) with word-level timing
- **Adjacency Pairs**: Conversational relationships between dialogue acts across speakers

## Files Created

### Main Processing Script
- `preprocess/dialogue_acts_process.py` - Main processing script
- `preprocess/scripts/dialogue_acts_process.sh` - SLURM batch script
- `preprocess/scripts/test_dialogue_acts.py` - Test script

### Supporting Files
- `preprocess/xml/ap-types.xml` - Adjacency pair type definitions

## Input Data Structure

The processing script expects the following XML files structure:

```
transcripts/
├── words/
│   └── ES2002a.A.words.xml          # Word-level transcripts with timing
├── dialogueActs/
│   └── ES2002a.A.dialog-act.xml     # Dialogue act annotations
├── adjacency-pairs/
│   └── ES2002a.adjacency-pairs.xml  # Conversational pair relationships
└── ontologies/
    ├── da-types.xml                 # Dialogue act type definitions
    └── ap-types.xml                 # Adjacency pair type definitions
```

## Output Data Structure

The script generates two CSV files:

### 1. Dialogue Acts CSV (`ami_dialogue_acts.csv`)
Contains word-level dialogue act annotations:

| Column | Description |
|--------|-------------|
| meeting_id | Meeting identifier (e.g., ES2002a) |
| speaker_id | Speaker identifier (A, B, C, D) |
| dact_id | Dialogue act instance ID |
| word | Word text or vocal sound |
| start_time | Word start time (seconds) |
| end_time | Word end time (seconds) |
| dialogue_act_type | Act type code (e.g., inf, sug, ass) |
| dialogue_act_gloss | Human-readable act description |
| dialogue_act_category | Act category (Minor, Task, Elicit, Other) |

### 2. Adjacency Pairs CSV (`ami_adjacency_pairs.csv`)
Contains conversational relationships:

| Column | Description |
|--------|-------------|
| meeting_id | Meeting identifier |
| pair_id | Adjacency pair instance ID |
| pair_type | Pair type code (e.g., apt_1, apt_3) |
| pair_type_gloss | Human-readable pair description |
| source_meeting_id | Source dialogue act meeting |
| source_speaker_id | Source dialogue act speaker |
| source_dact_id | Source dialogue act ID |
| target_meeting_id | Target dialogue act meeting |
| target_speaker_id | Target dialogue act speaker |
| target_dact_id | Target dialogue act ID |

## Dialogue Act Categories

The AMI corpus uses a hierarchical dialogue act taxonomy:

### Major Categories:
- **Minor**: Backchannel, Stall, Fragment
- **Task**: Inform, Suggest, Assess
- **Elicit**: Elicit-Inform, Elicit-Offer-Or-Suggestion, Elicit-Assessment, Elicit-Comment-Understanding
- **Other**: Offer, Comment-About-Understanding, Be-Positive, Be-Negative, Other

### Common Dialogue Acts:
- `inf` (Inform): Providing information
- `sug` (Suggest): Making suggestions
- `ass` (Assess): Making assessments
- `el.inf` (Elicit-Inform): Asking for information
- `bck` (Backchannel): Minimal responses
- `fra` (Fragment): Incomplete utterances

## Usage

### Basic Usage

```bash
python preprocess/dialogue_acts_process.py \
    --input "/path/to/transcripts" \
    --output "/path/to/output" \
    --include_adjacency_pairs
```

### SLURM Batch Processing

```bash
sbatch preprocess/scripts/dialogue_acts_process.sh
```

### Advanced Options

```bash
python preprocess/dialogue_acts_process.py \
    --input "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts" \
    --output "/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_dialogue_acts" \
    --dialogue_acts_dir "/path/to/dialogueActs" \
    --da_types_file "/path/to/da-types.xml" \
    --skip_adjacency_pairs  # Skip adjacency pairs processing
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input transcripts directory | `/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts` |
| `--output` | Output directory for CSV files | `/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_dialogue_acts` |
| `--dialogue_acts_dir` | Directory with dialogue act XML files | `{input}/dialogueActs` |
| `--da_types_file` | Dialogue act types definition file | `{input}/ontologies/da-types.xml` |
| `--include_adjacency_pairs` | Process adjacency pairs (default: True) | - |
| `--skip_adjacency_pairs` | Skip adjacency pairs processing | - |

## Testing

Run the test script to verify functionality:

```bash
python preprocess/scripts/test_dialogue_acts.py
```

This will test:
- Loading dialogue act types from XML
- Loading adjacency pair types from XML
- Word ID extraction from href references
- Dialogue act information extraction

## Integration with Existing Pipeline

The dialogue acts processing integrates with the existing disfluency/laughter processing:

```python
# Process disfluency and laughter
python preprocess/disfluency_laughter_process.py \
    --combine_events dialogue_acts \
    --input "/path/to/transcripts" \
    --output "/path/to/output"

# Process dialogue acts separately
python preprocess/dialogue_acts_process.py \
    --input "/path/to/transcripts" \
    --output "/path/to/output"
```

## Data Analysis Examples

### Python Analysis

```python
import pandas as pd

# Load dialogue acts data
da_df = pd.read_csv('ami_dialogue_acts.csv')

# Analyze dialogue act distribution by speaker
da_counts = da_df.groupby(['speaker_id', 'dialogue_act_type']).size().unstack(fill_value=0)
print(da_counts)

# Load adjacency pairs data
ap_df = pd.read_csv('ami_adjacency_pairs.csv')

# Analyze conversational patterns
pair_patterns = ap_df.groupby(['source_speaker_id', 'target_speaker_id']).size()
print(pair_patterns)
```

### Common Analysis Tasks

1. **Dialogue Act Frequency**: Which acts are most common per speaker?
2. **Conversational Patterns**: Who responds to whom most frequently?
3. **Temporal Analysis**: How do dialogue acts change over meeting time?
4. **Turn-Taking**: What's the typical adjacency pair structure?

## Troubleshooting

### Common Issues

1. **Missing XML Files**: Ensure all required XML files exist in expected locations
2. **Namespace Issues**: The script handles NITE namespaces automatically
3. **Memory Usage**: Large meetings may require more memory allocation
4. **File Permissions**: Ensure read access to input files and write access to output directory

### Error Messages

- `"Dialogue acts file not found"`: Check `--dialogue_acts_dir` path
- `"Adjacency pairs file not found"`: Check `adjacency-pairs` subdirectory exists
- `"Error loading dialogue act types"`: Verify `da-types.xml` format and path

## Performance

### Processing Times (Approximate)
- Single speaker: ~5-10 seconds
- Single meeting (4 speakers): ~30 seconds
- Full corpus (100+ meetings): ~1-2 hours

### Memory Requirements
- Small meetings: ~50MB RAM
- Large meetings: ~200MB RAM
- Full corpus: ~1-2GB RAM

## Related Documentation

- [Disfluency/Laughter Processing](../preprocess/README.md)
- [AMI Corpus Documentation](https://groups.inf.ed.ac.uk/ami/corpus/)
- [NITE XML Toolkit](http://nite.sourceforge.net/) 