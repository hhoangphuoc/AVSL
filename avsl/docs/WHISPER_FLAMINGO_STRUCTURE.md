# Directory Structure for Whisper-Flamingo Comprehensive Test

## Current Setup ✅

The directory structure is correctly configured as follows:

```
AVSL/
└── avsl/                                    # PROJECT_ROOT
    ├── test_whisper_flamingo.py            # Main test script
    ├── whisper_flamingo_ft_ami.py           # Training script
    ├── config/
    │   └── ami_whisper_flamingo_large.yaml  # Configuration file
    └── scripts/
        └── test/                            # SCRIPT_DIR
            └── run_comprehensive_test.sh    # Test runner script (you are here)
```

## How It Works

### Directory Calculation
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Result: /home/s2587130/AVSL/avsl/scripts/test

PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
# Result: /home/s2587130/AVSL/avsl (two levels up)
```

### Execution Flow
1. **Test runner starts** in: `/home/s2587130/AVSL/avsl/scripts/test/`
2. **Changes directory** to: `/home/s2587130/AVSL/avsl/` (PROJECT_ROOT)
3. **Runs test script**: `python test_whisper_flamingo.py config/ami_whisper_flamingo_large.yaml`
4. **All paths are relative** to PROJECT_ROOT

## How to Run

### Option 1: Using SLURM (Recommended)
```bash
cd /home/s2587130/AVSL/avsl/scripts/test
sbatch run_comprehensive_test.sh
```

### Option 2: Run test directly
```bash
cd /home/s2587130/AVSL/avsl
python test_whisper_flamingo.py config/ami_whisper_flamingo_large.yaml
```

## File Verification ✅

All required files are in the correct locations:

- ✅ Test runner: `/home/s2587130/AVSL/avsl/scripts/test/run_comprehensive_test.sh`
- ✅ Test script: `/home/s2587130/AVSL/avsl/test_whisper_flamingo.py`
- ✅ Training script: `/home/s2587130/AVSL/avsl/whisper_flamingo_ft_ami.py`
- ✅ Config file: `/home/s2587130/AVSL/avsl/config/ami_whisper_flamingo_large.yaml`

## Logs

Test logs will be saved to:
- Location: `/home/s2587130/AVSL/avsl/logs/`
- Format: `comprehensive_test_YYYYMMDD_HHMMSS.log`

## Success Output

If the directory structure is working correctly, you should see:
```
Script directory: /home/s2587130/AVSL/avsl/scripts/test
Project root: /home/s2587130/AVSL/avsl
Changed to project root: /home/s2587130/AVSL/avsl
Using config file: config/ami_whisper_flamingo_large.yaml
```

The test should then proceed to run all 9 test categories successfully. 