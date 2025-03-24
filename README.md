# AVSL
An Audio-Visual Speech Recognition (AVSR) design to transcribe disfluencies and laughter in conversational speech


## Dataset
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

```
The folders `audio_segments`, `video_segments` and `transcript_segments` are created to stored segmented **sentence-level** transcripts based on timestamps, audio and video cut segment corresponding to that timestamps.

### Transcripts
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
[meeting_id]
```



### Audio-Video Resource
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

## Preprocessing
### Transcript Description:
The task here is to combine word-level timestamps of sequence of words into sentence-level transcript (segment). 

We make use of two folders from original transcripts to do this: `transcripts/segments` and `transcripts/words` are used, which specify for **[meeting_id].[speaker_id]** segments and words.

For example, corresponding to speaker A in meeting EN2001a, the following files is used:

    - EN2001a.A.segments.xml
    - EN2001a.A.words.xml

and the file output will be outputting to: `transcript_segments`, which corresponding named: `[meeting_id]-[speaker_id].txt`. For example: **EN2001a-A.txt**. Each sentence of the output will be including start time, end time, and transcript sentence.

For example:

    [11.04-15.632] Does anyone want to see uh Steve's feedback from the specification?

### Process Transcript:
At each transcript file (e.g. `segments/EN2001a.A.segments.xml`, `words/EN2001a.A.words.xml`), the following will be processed:
- A sequence of words that concatenated into a sentence, each word specify by its id, for example: **EN2001a.A.words0**. These mostly are indicated in word tag: `<w .../>`
- Special word tag `<w/>`, i.e. punctuation and truncation are handled differently, in which:
    - Punctuation: `<w ... punc=true />` - attach with latest word without space
    - Truncation: `<w ... trunc=true />` - skip in the output transcript

- Special non-word tag also being annotated differently as followed:
    - Disfluent marker: `<disfmarker .../>` - skip in the output transcript
    - Laughter vocal sound: `<vocalsound type="laugh" />` - annotated as `<laugh>`
