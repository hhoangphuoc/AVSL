# Whisper-Flamingo Training Issues - Fixes Applied

This document summarizes the critical issues identified from your training logs and the fixes that have been implemented.

## 🔍 Issues Identified from Logs

### 1. **Video Object Type Mismatch**
**Problem**: HuggingFace datasets return `decord.video_reader.VideoReader` objects, not file paths.
**Evidence**: 
```
Warning: Could not extract path from video object: <class 'decord.video_reader.VideoReader'>
failed loading None (0 / 3)
```

### 2. **OpenCV Empty Filename Error**
**Problem**: `None` paths being passed to video loading functions.
**Evidence**:
```
OpenCV(4.5.4) ...cap_images.cpp:293: error: (-215:Assertion failed) !_filename.empty() in function 'open'
```

### 3. **Tokenizer Overflow Error**
**Problem**: Negative token IDs (like -100 for padding/ignore) being passed to tokenizer decoder.
**Evidence**:
```
OverflowError: can't convert negative int to unsigned
```

## ✅ Fixes Applied

### 1. **Enhanced Video Object Handling**

**File**: `avsl/utils_hf_video.py`
- Added support for `decord.VideoReader` objects
- New function: `load_video_feats_from_decord_reader()` 
- Reads video frames directly from VideoReader using `get_batch()`
- Handles RGB to grayscale conversion
- Applies proper normalization and cropping

**Key Changes**:
```python
# New function to handle decord.VideoReader objects directly
def load_video_feats_from_decord_reader(video_reader, train=False, **kwargs):
    # Get frames using video_reader.get_batch(frame_indices)
    # Convert RGB to grayscale, normalize, crop
    # Return [T, H, W, 1] format
```

### 2. **Improved Path Extraction**

**File**: `avsl/utils_hf_video.py`
- Enhanced `extract_video_path_from_hf_object()` to detect VideoReader objects
- Added fallback methods for different object types
- Better error handling and logging

### 3. **Fixed Tokenizer Overflow**

**File**: `avsl/whisper_flamingo_ft_ami.py` 
- Added filtering for negative token IDs before decoding
- Prevents overflow errors from padding tokens (-100)

**Before**:
```python
o_list.append(self.tokenizer.decode([t for t in o_tokens if t.item() not in self.special_token_set]))
```

**After**:
```python
valid_o_tokens = [t for t in o_tokens if t.item() >= 0 and t.item() not in self.special_token_set]
o_list.append(self.tokenizer.decode(valid_o_tokens))
```

### 4. **Enhanced Error Handling**

**File**: `avsl/whisper_flamingo_ft_ami.py`
- Added comprehensive try-catch blocks around video processing
- Better logging and debugging information
- Graceful fallback to dummy features when video loading fails

## 🧪 Testing the Fixes

### Quick Test
Run the targeted debugging script:
```bash
cd /home/s2587130/AVSL/avsl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper-flamingo
python debug_video_issue.py
```

This will test:
- Video object type detection
- Video feature loading from VideoReader objects  
- Tokenizer fix for negative token IDs

### Full Test
Run the comprehensive dataset test:
```bash
python test_hf_dataset.py
```

## 🚀 Running Training

### 1. **Test First** (Recommended)
```bash
python debug_video_issue.py
```
Ensure all tests pass before proceeding.

### 2. **Run Training**
```bash
sbatch scripts/train/whisper_flamingo_ft.sh
```

### 3. **Monitor Logs**
Look for these positive indicators:
```
✓ HuggingFace Video utilities imported successfully
✓ Detected decord.VideoReader object
✓ Video features loaded successfully!
✓ Video features contain actual data
```

## 📋 Expected Behavior Changes

### Before Fixes:
- ❌ Video loading failed with "failed loading None"
- ❌ Training crashed with tokenizer overflow
- ❌ OpenCV errors from empty filenames

### After Fixes:
- ✅ VideoReader objects processed directly
- ✅ Video features loaded successfully
- ✅ Tokenizer handles negative tokens gracefully
- ✅ Training proceeds without crashes

## 🔧 Files Modified

1. **`avsl/utils_hf_video.py`** - Enhanced video object handling
2. **`avsl/whisper_flamingo_ft_ami.py`** - Fixed tokenizer and video processing
3. **`avsl/test_hf_dataset.py`** - Updated test script
4. **`avsl/debug_video_issue.py`** - New targeted debugging script

## 🎯 Key Improvements

- **Robust Video Loading**: Handles multiple HF video object types
- **Direct VideoReader Support**: No need for temporary files
- **Tokenizer Safety**: Filters out problematic token IDs
- **Better Debugging**: Comprehensive test scripts and logging
- **Graceful Degradation**: Training continues with dummy features if needed

## ⚠️ Notes

- The fixes maintain backward compatibility
- Dummy features are used as fallback (training will continue but may have reduced performance)
- All improvements follow AVSL project conventions
- Error handling is comprehensive but non-intrusive

## 🆘 If Issues Persist

1. Run `debug_video_issue.py` and share the output
2. Check that `decord` package is installed in your environment
3. Verify dataset paths and permissions
4. Monitor GPU memory usage (reduce batch size if needed)

The implemented fixes should resolve the specific errors seen in your training logs and allow the whisper-flamingo training to proceed successfully. 