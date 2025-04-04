# Preprocessing 

This preprocess details corresponding to the files used in `AVSL/preprocess/` folder, including

## Processing Transcript
### Transcript Description
The task here is to combine word-level timestamps of sequence of words into sentence-level transcript (segment). 

We make use of two folders from original transcripts to do this: `transcripts/segments` and `transcripts/words` are used, which specify for **[meeting_id].[speaker_id]** segments and words.

For example, corresponding to speaker A in meeting EN2001a, the following files is used:

    - EN2001a.A.segments.xml
    - EN2001a.A.words.xml

and the file output will be outputting to: `transcript_segments`, which corresponding named: `[meeting_id]-[speaker_id].txt`. For example: **EN2001a-A.txt**. Each sentence of the output will be including start time, end time, and transcript sentence.

For example:

    [11.04-15.632] Does anyone want to see uh Steve's feedback from the specification?

### Process Transcript
At each transcript file (e.g. `segments/EN2001a.A.segments.xml`, `words/EN2001a.A.words.xml`), the following will be processed:
- A sequence of words that concatenated into a sentence, each word specify by its id, for example: **EN2001a.A.words0**. These mostly are indicated in word tag: `<w .../>`
- Special word tag `<w/>`, i.e. punctuation and truncation are handled differently, in which:
    - Punctuation: `<w ... punc=true />` - attach with latest word without space
    - Truncation: `<w ... trunc=true />` - skip in the output transcript

- Special non-word tag also being annotated differently as followed:
    - Disfluent marker: `<disfmarker .../>` - skip in the output transcript
    - Laughter vocal sound: `<vocalsound type="laugh" />` - annotated as `<laugh>`

### Process Disfluency & Laughter
This additional process mainly to extract only the  disfluency and laughter occurences of words/vocalsound in all the meetings.

The output of this process will be a csv file located at `dsfl_laugh/disfluency_laughter_marker.csv`, contains the following columns:

```bash
- meeting_id #(e.g. ES2002a) 
- speaker_id #(e.g. B)
- word #float
- starttime #float
- endtime
- disfluency_type #one of 19 AMI disfluency types
- islaugh #(1 if it is laughter, else 0)
```

**Disfluency**

Disfluency is process by matching the disfluency types, annotated at `transcripts/ontologies` with the disfluency word which specify in `transcripts/disfluency` folder, while the actual word located at `transcripts/words`.

**Laughter**

Instead tagged as `<vocalsound ...type="laugh"/>` in `transcripts/words`, which annotated as `<laugh>` in the csv.

**Summary of the file:** 

Using the following command to discover overall statistic
```bash
echo "Total entries:" &&  wc -l dsfl_laugh/disfluency_laughter_markers.csv &&  echo "Laugh entries:" &&  grep -c ,,1 dsfl_laugh/disfluency_laughter_markers.csv &&  echo "Disfluency entries:" &&  grep -v ,,1 dsfl_laugh/disfluency_laughter_markers.csv |  grep -v disfluency_type |  wc -l
```
It gives the following result:
```bash
Total entries:
    60479 
    dsfl_laugh/disfluency_laughter_markers.csv
Laugh entries:
    16524
Disfluency entries:
    43954
```

---
## Segmenting Audio and Video Sources (AMI)

Based on the `data/transcript_segments`, which contains the transcript for each `[meeting_id]-[speaker_id].txt` sentence-level transcript, We use the marked timestamp to segment the original audio and video sources into multiple sources.
- **Audio:** Located in `data/amicorpus/[meeting_id]/audio`, where `speaker_id` annotated as: Headset-[0-4] corresponding to speaker A-E
- **Video:** Located in `data/amicorpus/[meeting_id]/video`, where `speaker_id` annotated as: Closeup[1-5] corresponding to speaker A-E

therefore, we mapping the audio-video naming with corresponding `speaker_id`, before segmenting. The audio and video segments are stored in `data/audio_segments` and `data/video_segments` respectively, with the naming convention as:
```bash
[meeting_id]-[speaker_id]-[starttime]-[endtime]-[src_type].[src_format]
```
in which: `src_type` and `src_format` corresponding to audio(.wav) and video(.mp4) respectively

### HuggingFace Dataset
The segmented audio and video are store in a single `HuggingFace Dataset`, in which the audio and video are wrapped in `Audio()` and `Video()` objects.

The dataset features are formatted:
```python

```

---
## Extracting Lip Video from Segmented Video