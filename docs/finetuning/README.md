# Fine-tuning Overview

This folder collects recipes for adapting different model variants to task-specific datasets.

| Model | Description | File |
|-------|-------------|------|
| Whisper-Flamingo | Audio–visual fusion model combining Whisper (audio) and AV-HuBERT (video) | [WHISPER_FLAMINGO_SETUP.md](mdc:docs/finetuning/WHISPER_FLAMINGO_FINETUNING.md) |
<!-- | AV-HuBERT S2S | Sequence-to-sequence fine-tuning of AV-HuBERT with a Transformer decoder | [SEQ2SEQ_FINETUNING.md](mdc:docs/finetuning/SEQ2SEQ_FINETUNING.md) | -->

Start with `Whisper-Flamingo` to start the fine-tuning with minimal engineering effort.
<!-- Move to the S2S pipeline if you need an end-to-end encoder-decoder architecture or plan to perform speech-to-speech translation.  -->