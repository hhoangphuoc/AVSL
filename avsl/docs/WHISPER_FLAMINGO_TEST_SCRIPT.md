# Comprehensive Whisper-Flamingo AMI Fine-tuning Test Suite

This test suite verifies that all components of your `whisper_flamingo_ft_ami.py` training script are working correctly, including the successful AV-HuBERT task registration fix.

## Quick Start

### Option 1: Run with the test runner script (Recommended)
```bash
cd /home/s2587130/AVSL/avsl
./run_comprehensive_test.sh
```

### Option 2: Run the test script directly
```bash
cd /home/s2587130/AVSL/avsl
python test_whisper_flamingo.py config/ami_whisper_flamingo_large.yaml
```

## What Gets Tested

The comprehensive test suite includes 9 major test categories:

### 1. **System Requirements** 
- ✅ PyTorch and CUDA availability
- ✅ Directory structure verification
- ✅ Required files existence check
- ✅ Model weights availability

### 2. **Import Tests**
- ✅ All required Python packages
- ✅ Whisper-Flamingo modules
- ✅ Fairseq with AV-HuBERT fix
- ✅ PyTorch Lightning components

### 3. **Model Loading Tests**
- ✅ Basic Whisper model loading
- ✅ Whisper-Flamingo with AV-HuBERT configuration
- ✅ Exact same parameters as training script
- ✅ AV-HuBERT task registration fix verification

### 4. **Dataset Loading Tests**
- ✅ AMI HuggingFace datasets (train/val/test)
- ✅ Dataset structure validation
- ✅ Required fields verification
- ✅ Audio and video data accessibility

### 5. **AmiVideoHFDataset Tests**
- ✅ Dataset class instantiation
- ✅ Sample retrieval and processing
- ✅ Audio resampling and padding
- ✅ Video feature loading
- ✅ Transcript tokenization

### 6. **Training Script Components**
- ✅ WhisperFlamingoModule import
- ✅ AmiVideoHFDataset class
- ✅ Training script integration

### 7. **Configuration Loading**
- ✅ YAML configuration parsing
- ✅ Parameter validation
- ✅ Model file path verification

### 8. **WhisperFlamingoModule Creation**
- ✅ Lightning module instantiation
- ✅ Model architecture verification
- ✅ Tokenizer setup
- ✅ Dataset length calculation

### 9. **Data Loaders Tests**
- ✅ Collation function
- ✅ Batch sampling strategy
- ✅ DataLoader components

## Expected Output

### Success Case
```
🎉 ALL TESTS PASSED!
✅ Your whisper_flamingo_ft_ami.py is ready for training!
✅ AV-HuBERT task registration fix is working correctly!

Next steps:
1. Ensure you have sufficient GPU resources
2. Verify your SLURM configuration (if using)
3. Run the training script:
   cd /home/s2587130/AVSL/avsl
   python whisper_flamingo_ft_ami.py config/ami_whisper_flamingo_large.yaml
```

### Partial Success Case
```
⚠️ AV-HuBERT specific issues found:
  • Model Loading Tests

🎯 CONCLUSION:
Training should work with fallback configuration (no AV-HuBERT).
To use AV-HuBERT, address the specific issues above.

To start training:
  cd /home/s2587130/AVSL/avsl
  python whisper_flamingo_ft_ami.py config/ami_whisper_flamingo_large.yaml
```

## Key Features

### AV-HuBERT Task Registration Fix Applied
The test script applies the same fix that worked in `test_av_hubert_fix.py`:
- ✅ Correct path setup for `avhubert_user_dir`
- ✅ Dummy argument workaround for fairseq registration
- ✅ Pre-import fairseq modules correctly
- ✅ Use exact same approach as training script

### Comprehensive Error Diagnosis
The test provides detailed diagnostic information:
- **Critical failures**: Will prevent training entirely
- **AV-HuBERT failures**: May allow fallback training
- **Minor failures**: Usually won't affect training

### Automatic Logging
When using the runner script:
- All output is logged to `logs/comprehensive_test_TIMESTAMP.log`
- Easy error filtering with provided commands
- Full test results preserved for debugging

## Troubleshooting

### Common Issues and Solutions

#### 1. "Cannot register duplicate model (av_hubert)"
**Status**: Should be fixed with this test suite
**If still occurs**: The fix needs adjustment - report this issue

#### 2. "Test dataset not found"
```bash
# Check if AMI datasets exist
ls -la /home/s2587130/AVSL/data/ami/av_hubert/
```
**Solution**: Ensure AMI preprocessing completed successfully

#### 3. "AV-HuBERT weights not found"
```bash
# Check model file
ls -la /home/s2587130/AVSL/avsl/models/large_noise_pt_noise_ft_433h_only_weights.pt
```
**Solution**: Download or verify the AV-HuBERT checkpoint file

#### 4. "CUDA not available"
**Impact**: Training will be very slow on CPU
**Solution**: Check GPU availability and CUDA installation

#### 5. Import errors
**Check**: 
- Python environment setup
- Required packages installation
- Whisper-flamingo repository cloning

## Files Created/Modified

1. **`test_whisper_flamingo.py`** - Main comprehensive test script
2. **`run_comprehensive_test.sh`** - Test runner with logging
3. **`COMPREHENSIVE_TEST_README.md`** - This documentation

## Advanced Usage

### Testing with Different Configurations
```bash
# Test with a specific config file
./run_comprehensive_test.sh config/custom_config.yaml

# Test with fallback configuration
python test_whisper_flamingo.py
```

### Viewing Detailed Logs
```bash
# View full log
cat logs/comprehensive_test_TIMESTAMP.log

# View only errors
grep -E '(✗|❌|💥|ERROR|Failed)' logs/comprehensive_test_TIMESTAMP.log

# View only successes
grep '✓' logs/comprehensive_test_TIMESTAMP.log
```

### Integration with CI/CD
The test script returns appropriate exit codes:
- `0`: All tests passed
- `1`: Some tests failed but training may still work
- Non-zero: Critical failures

## Next Steps After Testing

1. **If all tests pass**: Run the training script directly
2. **If AV-HuBERT tests fail**: Training will work with fallback configuration
3. **If critical tests fail**: Fix the identified issues before training

## Support

If you encounter issues:
1. Run the comprehensive test with logging
2. Check the specific error messages in the log
3. Follow the troubleshooting guidance provided
4. Review the original AV-HuBERT fix documentation in `AV_HUBERT_FIX_README.md`

The test suite is designed to catch issues before training starts, saving time and computational resources. 