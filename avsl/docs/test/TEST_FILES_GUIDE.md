# Test Files Guide for AVSL Project

## 🧪 **Overview**

This guide documents the comprehensive test suite for the AVSL whisper-flamingo training project. All test files have been standardized with consistent imports and path handling.

## 📁 **Test File Structure**

```
avsl/test/
├── test_hf_dataset_comprehensive.py  # 🆕 MAIN HF dataset test (merged & enhanced)
├── test_hf_dataset.py                # Original HF dataset test (basic)
├── test_video_validation.py          # Robust video validation & corruption detection
├── debug_video_issue.py              # Video loading & tokenizer debugging
├── test_whisper_flamingo.py          # Complete training pipeline test
└── test_av_hubert_fix.py             # AV-HuBERT registration fix test
```

## 🎯 **Test Files Descriptions**

### 1. **`test_hf_dataset_comprehensive.py`** ⭐ **RECOMMENDED**
**Purpose**: Comprehensive HuggingFace dataset testing (merged from old versions)

**What it tests**:
- ✅ Dataset structure and loading from multiple sources
- ✅ Video processing (both basic and robust methods)
- ✅ Robust video filtering and corruption detection
- ✅ AmiVideoHFDataset creation and sample loading
- ✅ Dataset format compatibility across different splits

**When to use**: 
- Primary test for dataset-related issues
- Before starting training to ensure dataset compatibility
- To diagnose video loading problems

**Usage**:
```bash
cd avsl
python test/test_hf_dataset_comprehensive.py
```

### 2. **`test_video_validation.py`**
**Purpose**: Dedicated robust video validation and corruption detection

**What it tests**:
- ✅ Video corruption detection across entire dataset
- ✅ Creates reports of corrupted files
- ✅ Tests video loading robustness
- ✅ Generates clean datasets for training

**When to use**:
- When you suspect corrupted videos in dataset
- To create comprehensive corruption reports
- Before large-scale training runs

**Usage**:
```bash
python test/test_video_validation.py
```

### 3. **`debug_video_issue.py`**
**Purpose**: Quick diagnostic test for specific video and tokenizer issues

**What it tests**:
- ✅ Video object handling (decord.VideoReader)
- ✅ Video path extraction
- ✅ Tokenizer negative token handling fix

**When to use**:
- Quick debugging of video loading issues
- Testing specific fixes for decord.VideoReader problems
- Verifying tokenizer overflow fixes

### 4. **`test_hf_dataset.py`**
**Purpose**: Basic HuggingFace dataset testing (original version, updated imports)

**What it tests**:
- ✅ Basic dataset loading and structure
- ✅ Video processing with robust error handling
- ✅ AmiVideoHFDataset creation

**When to use**:
- Simple dataset verification
- When you need basic functionality without comprehensive testing

### 5. **`test_whisper_flamingo.py`**
**Purpose**: Complete training pipeline verification

**What it tests**:
- ✅ All training script imports
- ✅ Model loading (with and without AV-HuBERT)
- ✅ Dataset loading and processing
- ✅ AmiVideoHFDataset integration
- ✅ Configuration loading
- ✅ Training component compatibility

**When to use**:
- Before starting training to verify entire pipeline
- To test model loading and configuration
- Complete system integration testing

### 6. **`test_av_hubert_fix.py`**
**Purpose**: AV-HuBERT task registration fix verification

**What it tests**:
- ✅ AV-HuBERT model loading without registration conflicts
- ✅ Multiple model loading (duplicate registration test)
- ✅ Different model configurations

**When to use**:
- When encountering "Cannot register duplicate model" errors
- Testing AV-HuBERT integration fixes

## 🚀 **Quick Start Testing**

### **Run All Tests** (Recommended)
```bash
cd avsl
sbatch scripts/test/run_all_tests.sh
```

### **Quick Dataset Check**
```bash
python test/test_hf_dataset_comprehensive.py
```

### **Corruption Detection**
```bash
python test/test_video_validation.py
```

### **Training Pipeline Check**
```bash
python test_whisper_flamingo.py config/ami_whisper_flamingo_large.yaml
```

## 🔧 **Import Structure (Standardized)**

All test files now use consistent path handling:

```python
# Consistent path setup (same across all test files)
current_dir = os.path.dirname(os.path.abspath(__file__))  # avsl/test
parent_dir = os.path.dirname(current_dir)  # avsl
project_root = os.path.dirname(parent_dir)  # AVSL
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo')
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert')

# Add to Python path (consistent with all test files)
sys.path.insert(0, project_root)
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)
sys.path.insert(0, parent_dir)  # For utils_hf_video import
```

## 📊 **Test Output Structure**

All tests create output in:
```
avsl/output/test_scripts/
├── comprehensive_test_summary.json      # Overall test results
├── video_filtering_test_results.json    # Video validation results
├── corrupted_videos_report.json         # Corruption details
└── full_dataset_corrupted_videos_report.json  # Full dataset analysis
```

## ⚠️ **Common Issues and Solutions**

### **Import Errors**
```bash
❌ Could not import HF Video utilities
```
**Solution**: Use `test_hf_dataset_comprehensive.py` which has fallback implementations

### **Video Loading Failures**
```bash
RuntimeError: Error reading video file
```
**Solution**: Run `test_video_validation.py` to identify and filter corrupted videos

### **AV-HuBERT Registration Conflicts**
```bash
Cannot register duplicate model
```
**Solution**: Run `test_av_hubert_fix.py` to verify the fix is working

### **Dataset Path Issues**
```bash
Dataset path does not exist
```
**Solution**: Check paths in test files and update for your environment

## 🎉 **Success Indicators**

### **All Tests Pass**
```
🎉 TESTS COMPLETED SUCCESSFULLY! Your whisper-flamingo training is ready!
```

### **Most Tests Pass (Expected)**
```
✅ Most tests passed! Your dataset should work for training.
⚠️ Some issues detected, but they may be expected (e.g., corrupted videos).
```

## 📝 **Test Logs**

- **SLURM Output**: `logs/whisper_flamingo_tests_*.log`
- **SLURM Errors**: `logs/whisper_flamingo_tests_*.err`
- **Test Results**: `output/test_scripts/*.json`

## 💡 **Best Practices**

1. **Always run comprehensive tests before training**
2. **Check corruption reports and clean datasets**
3. **Use fallback implementations when utils aren't available**
4. **Monitor test outputs for specific guidance**
5. **Update dataset paths for your environment**

## 🔗 **Related Documentation**

- `ROBUST_VIDEO_HANDLING_GUIDE.md` - Video handling improvements
- `FIXES_SUMMARY.md` - Summary of all fixes applied
- `whisper_flamingo_ft_ami.py` - Main training script
- `utils_hf_video.py` - Video utilities

---

**Quick Command Reference**:
```bash
# Run all tests
sbatch scripts/test/run_all_tests.sh

# Comprehensive dataset test
python test/test_hf_dataset_comprehensive.py

# Video validation
python test/test_video_validation.py

# Training pipeline test  
python test_whisper_flamingo.py config/ami_whisper_flamingo_large.yaml
``` 