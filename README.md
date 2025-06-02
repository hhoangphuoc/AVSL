# AVSL
An Audio-Visual Speech Recognition (AVSR) design to transcribe disfluencies and laughter in conversational speech

## Features
- Multi-modal AVHuBERT model for speech recognition
- Sequence-to-sequence fine-tuning with encoder-decoder architecture
- [YAML-based configuration system](docs/YAML_CONFIG.md) for flexible model parameterization
- Support for different modality fusion strategies (concatenation, sum, weighted sum)
- Robust handling of missing modalities

## Project Structure
- **`avsl/`**: Workflow and core modules for fine-tuning and inference of the Whisper Flamingo model.
- **`preprocess/`**: Comprehensive data preprocessing pipelines (audio, video, lip-video, transcript) converting the AMI Meeting Corpus into Hugging Face datasets.
- **`whisper_flamingo/`**: Original implementation of the Whisper Flamingo multimodal model.
- **`data/`**, **`pretrained/`**, **`checkpoints/`**, **`logs/`**, **`output/`**: Data storage, model weights, logs, and experiment outputs.
- **`scripts/`**, **`utils/`**, **`docs/`**, **`resources/`**, **`cache/`**: Utilities, documentation, static resources, and cached metrics.

# Dataset
We use AMI Meeting Corpus as the main dataset, including both audio and video sources from several AMI Meetings. The dataset stored at:
```bash
/deepstore/datasets/hmi/speechlaugh-corpus/ami

ami/
    |_transcripts
        |_segments / {words | disfluency | ...}
    |
    |_amicorpus
    |_...
    |_ audio_segments
    |_ video_segments
    |_ transcript_segments
    |_ dsfl 
        |_ disfluency_laughter_marker.csv #marking position of disfluency + laughter occurence (for detection purpose)
    |_ ami_laughter
        |_ ami_laugh_markers.csv #marking position of laughter +fluent speech occurrence (for detection purpose)
        |_ segments
            |_ audio_segments
            |_ video_segments
            |_ lip_segments

```
The folders `audio_segments`, `video_segments` and `transcript_segments` are created to stored segmented **sentence-level** transcripts based on timestamps, audio and video cut segment corresponding to that timestamps.

The child folder under: `ami_laughter/segments/...` are storing segmented **word-level** events of either laughter or fluent speech events, based on timestamps of such event and the corresponding audio and video (if exist)

## Transcripts
```bash
path = ami/transcripts/[specific_folder]
```
contains original transcripts of different type of annotations, including:
-  `words`, 
- `segments`-timestamp, position and sequence of words belong to each transcripts segment (corresponding to single speaker) 
- `disfluency`- timestamp, types and position of whether the disfluencies occur
- `ontologies` - contains different files that specified different annotation types: disfluencies, dialog acts, postures, etc.
- ...

In segments folders `ami/transcripts/segments`, it contains all the segments of every meetings and every person in the meeting, with the naming convention:
```
[meeting_id].[speaker_id].segments.xml
```

The processed transcript segments are stored in `ami/transcript_segments` folder, contains all the sentence-by-sentence transcripts of each speaker in each meeting, including the transcript (text), and the start and end timestamp, each file named with the convention:
```
[meeting_id]-[speaker_id].txt
```



## Audio-Video Resource
```
path = ami/amicorpus/[meeting_id]
```
contains various folders corresponding to various meetings (`meeting_id`), each including 2 folders:
- `audio`: contains 4 seperate audio from 4 different speaker, naming as followed:
    ```
    [`meeting_id`].Headset-[`speaker_id`].wav 
    #speaker_id: 0,1,2,3
    ```
- `video`: contains 4 closeup video corresponding to 4 speakers + 1 corner video + 1 overhead video. 

    The naming convention for each video is follow:
        
    ```
    [`meeting_id`].Closeup[`speaker_id`].avi 
    #speaker_id: 1,2,3,4
    ```

---
<br>

# Data Processing
## [Preprocessing](docs/preprocess/Preprocess.md)
The preprocessing part described the details process in converting the original AMI Meeting Corpus into segmented audio-video with the corresponding transcripts, specified to every `[meeting_id]-[speaker_id]`.This process handling the processing for two modalities: audio and video, separately. 

Additionally, `disfluency` and `laughter` events with specific timestamps and audio/video segments extracted from original AMI has also been processed separately. This is designed for whom interesting in detecting paralinguistic events in AMI Meeting Corpus.



## [HuggingFace Dataset](docs/preprocess/DatasetProcess.md)
The preprocessed data also stored in `HuggingFace Dataset`, where the audio and video have been additionally reformatted and structurized. See the documentation for details approach.

<!-- # Fine-tuning AVHuBERT

## Configuration

The AVHuBERT model can be configured using YAML configuration files. See the [YAML Configuration Documentation](docs/YAML_CONFIG.md) for details.

## Sequence-to-Sequence Fine-tuning

For fine-tuning AVHuBERT with a sequence-to-sequence approach, use the `finetune_avhubert_seq2seq.py` script:

```bash
python finetune_avhubert_seq2seq.py \
  --config_yaml config/avhubert_large.yaml \
  --output_dir output/avhubert_ft \
  --dataset_name ami_corpus \
  --do_train \
  --do_eval
```

## Example Scripts

Example scripts for fine-tuning are provided in the `scripts/` directory:

- `scripts/finetune_avhubert_with_yaml.sh`: Fine-tuning with a YAML configuration
-->
# Fine-tuning an Audio-Visual Model
## Quickstart
Follow this 5-step recipe to go from a fresh clone to a fully trained audio-visual speech model.    

1. **Install the environment**  
    See the detailed guide in [docs/setup](mdc:docs/setup) or run:
    ```bash
    conda env create -f environment.yml  # or python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```
    Optionally install Fairseq in editable mode (needed by AV-HuBERT & Flamingo):
    ```bash
    pip install --editable av_hubert/fairseq  # root copy
    ```

2. **Download datasets & models**  
    Prepare the AMI corpus under `data/ami/`
    
    And fetch pre-trained models checkpoints, from original model:
    ```bash
    conda activate environment
    cd avsl
    sbatch scripts/preparation/download_models.sh
    ```

3. **Pre-process the data**  
    Convert raw AMI transcripts / audio / video to sentence-level segments and HF datasets:
    ```bash
    sbatch preprocess/scripts/ami_process.sh
    ```
    (See [docs/preprocess](docs/preprocess) for a deep dive.)

4. **Fine-tune the model**  

    Test the scripts to make sure the model configuration works correctly:
    ```bash
    cd avsl
    sbatch scripts/test/run_all_tests.sh
    ```

    Once the test run completely, run the fine-tuning of Whisper-Flamingo + AV-HuBERT fusion, using:
    ```bash
    cd avsl
    sbatch scripts/train/whisper_flamingo_ft.sh config/ami_whisper_flamingo_large.yaml
    ```


5. **Evaluate / Inference**  
   ```bash
   cd avsl
   python scripts/eval_whisper_flamingo.py --ckpt checkpoints/whisper_flamingo_ft/model.pt
   ```
