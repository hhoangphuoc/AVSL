# AVSL Technical Information

AVSL (Audio-Visual Speech Recognition) is a project for transcribing disfluencies and laughter in conversational speech using multi-modal deep learning models.

## Project Structure

- `models/`: Contains the model implementations
  - `av_hubert_model.py`: AV-HuBERT model implementation
  - `av_whisper_gated_model.py`: Whisper-based model with gating mechanism

- `modules/`: Contains modular components of the models
  - `av_hubert_encoder.py`: Encoder for processing audio and visual inputs
  - `av_hubert_decoder.py`: Decoder for generating text outputs
  - `resnet.py`: ResNet implementation for visual feature extraction

- `config/`: Configuration files
  - `av_hubert_config.py`: Configuration class for the AV-HuBERT model

- `preprocess/`: Scripts for data preprocessing
  - `dataset_process.py`: Main script for preparing the dataset
  - `audio_process.py`: Audio processing utilities
  - `video_process.py`: Video processing utilities

- `runnable/`: Shell scripts for running preprocessing and training tasks
  - `ami_process.sh`: Process AMI meeting corpus
  - `lip_process.sh`: Extract lip regions from videos
  - `dsfl_process.sh`: Process disfluency data

## Primary Model Architecture

The project implements an Audio-Visual HuBERT (AV-HuBERT) model for speech recognition that can process both audio and visual (lip movement) inputs. Key components:

1. **AVHuBERTModel**: Base model that combines audio and visual encoders
2. **AVHuBERTForCTC**: Implementation using CTC loss for ASR tasks
3. **AVHuBERTForSpeech2Text**: Sequence-to-sequence implementation for speech-to-text
4. **AVHuBERTForConditionalGeneration**: For conditional text generation

The model can operate in audio-only, visual-only, or audio-visual fusion modes with different fusion strategies (concatenation, addition).

## Data Pipeline

1. **Preprocessing**: Segment audio/video from AMI Meeting Corpus based on transcript timestamps
2. **Feature Extraction**: Extract audio features and lip region features
3. **Dataset Creation**: Create HuggingFace datasets for training
4. **Training**: Fine-tune the AV-HuBERT model on the preprocessed data

## Running the Code

1. **Setup Environment**: `pip install -r requirements.txt`
2. **Preprocess Data**: `python preprocess/dataset_process.py --mode [mode] --source_dir [source_dir] --dataset_path [output_path]`
3. **Training**: `python finetune_avhubert.py --model_name_or_path [path] --config_name [path] --dataset_name [name] --output_dir [dir]`