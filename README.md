# AVSL
An Audio-Visual Speech Recognition (AVSR) design to transcribe disfluencies and laughter in conversational speech

## Features
- Multi-modal AVHuBERT model for speech recognition
- Sequence-to-sequence fine-tuning with encoder-decoder architecture
- [YAML-based configuration system](docs/YAML_CONFIG.md) for flexible model parameterization
- Support for different modality fusion strategies (concatenation, sum, weighted sum)
- Robust handling of missing modalities


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
    |_ dsfl_laugh 
        |_ disfluency_laughter_marker.csv #marking position of disfluency + laughter occurence (for detection purpose)

```
The folders `audio_segments`, `video_segments` and `transcript_segments` are created to stored segmented **sentence-level** transcripts based on timestamps, audio and video cut segment corresponding to that timestamps.

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
## [Preprocessing](docs/Preprocess.md)
The preprocessing part described the details process in converting the original AMI Meeting Corpus into segmented audio-video with the corresponding transcripts, specified to every `[meeting_id]-[speaker_id]`.This process handling the processing for two modalities: audio and video, separately. 

Additionally, `disfluency` and `laughter` events with specific timestamps and audio/video segments extracted from original AMI has also been processed separately. This is designed for whom interesting in detecting paralinguistic events in AMI Meeting Corpus.



## [HuggingFace Dataset](docs/DatasetProcess.md)
The preprocessed data also stored in `HuggingFace Dataset`, where the audio and video have been additionally reformatted and structurized. See the documentation for details approach.

# Fine-tuning AVHuBERT

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

See [SEQ2SEQ_FINETUNING.md](docs/SEQ2SEQ_FINETUNING.md) for detailed instructions and options.

## Example Scripts

Example scripts for fine-tuning are provided in the `scripts/` directory:

- `scripts/finetune_avhubert_with_yaml.sh`: Fine-tuning with a YAML configuration
