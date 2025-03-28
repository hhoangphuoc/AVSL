# AVSL
An Audio-Visual Speech Recognition (AVSR) design to transcribe disfluencies and laughter in conversational speech


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

# [Preprocessing](docs/Preprocess.md)
The preprocessing part described the details process in converting the original AMI Meeting Corpus into segmented audio-video with the corresponding transcripts, specified to every `[meeting_id]-[speaker_id]`.

The preprocessed data also stored in `HuggingFace Dataset`, where the audio and video have their own feature decoded. 

