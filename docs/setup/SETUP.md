# Setup & Installation Guide

This guide walks you through creating a working environment for AVSL on Linux or an HPC cluster.

## 1. Prerequisites
- Python 3.10+ (tested on 3.10)
- CUDA-capable GPU and the matching CUDA toolkit version for your PyTorch build
- Conda or virtualenv for managing Python packages
- AMI Meeting Corpus (see Data section below)

## 2. Clone the Repository
```bash
git clone https://github.com/your-org/AVSL.git
cd AVSL
```

## 3. Create the Environment
```bash
# Option A: Conda
conda env create -f environment.yml   # provides python + torch + deps
conda activate avsl

# Option B: venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Install Fairseq (required by AV-HuBERT & Flamingo)
Two Fairseq copies live in the repo (root and whisper_flamingo). Installing either one in editable mode is enough:
```bash
pip install -e av_hubert/fairseq       # root copy
# or, if you prefer the Flamingo fork
pip install -e whisper_flamingo/av_hubert/fairseq
```
> On some systems you may need to pin `numpy<=1.23` to silence deprecated alias warnings used inside Fairseq.

## 5. Download Pre-trained Weights
```bash
bash scripts/download_models.sh
```
Weights are stored under `pretrained/` and automatically resolved by the training scripts.

## 6. Prepare the Dataset
The project expects the AMI corpus under `data/ami/`. Follow the preprocessing guide:
```bash
python preprocess/scripts/run_preprocess.py --config preprocess/config/ami.yaml
```

## 7. Quick Sanity Check
```bash
python -c "import fairseq, whisper_flamingo, avsl; print('✓ setup ok')"
```

## 8. Next Steps
- Fine-tune a model: see [docs/finetuning](mdc:docs/finetuning)
- Explore preprocessing internals: see [docs/preprocess](mdc:docs/preprocess)
- Dive into code structure: refer to the Project Structure section in the root README. 