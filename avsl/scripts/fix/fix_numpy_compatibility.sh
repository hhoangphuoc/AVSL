#!/bin/bash

# Fix NumPy compatibility issues in fairseq
# This script fixes the deprecated np.float, np.int, np.complex, np.bool, np.object, np.str usage

set -e  # Exit on any error

echo "=== Fixing NumPy Compatibility Issues in Fairseq ==="

# Navigate to fairseq directory
FAIRSEQ_DIR="/home/s2587130/AVSL/whisper_flamingo/av_hubert/fairseq"

if [ ! -d "$FAIRSEQ_DIR" ]; then
    echo "Error: Fairseq directory not found at $FAIRSEQ_DIR"
    exit 1
fi

cd "$FAIRSEQ_DIR"

echo "Fixing NumPy deprecated aliases in fairseq..."

# Create backup directory
BACKUP_DIR="numpy_fix_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# List of files that commonly have NumPy compatibility issues
FILES_TO_FIX=(
    "fairseq/data/indexed_dataset.py"
    "fairseq/data/plasma_utils.py"
    "fairseq/data/huffman/huffman_coder.py"
)

# Function to fix a file
fix_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        
        # Create backup
        cp "$file" "$BACKUP_DIR/$(basename $file).backup"
        
        # Apply fixes
        sed -i 's/np\.float\b/np.float64/g' "$file"
        sed -i 's/np\.int\b/np.int64/g' "$file"
        sed -i 's/np\.complex\b/np.complex128/g' "$file"
        sed -i 's/np\.bool\b/bool/g' "$file"
        sed -i 's/np\.object\b/object/g' "$file"
        sed -i 's/np\.str\b/str/g' "$file"
        
        echo "✓ Fixed $file"
    else
        echo "⚠ File not found: $file"
    fi
}

# Fix known problematic files
for file in "${FILES_TO_FIX[@]}"; do
    fix_file "$file"
done

# Search for and fix any remaining instances
echo "Searching for remaining NumPy deprecated aliases..."

# Find all Python files and fix them
find . -name "*.py" -type f -exec grep -l "np\.\(float\|int\|complex\|bool\|object\|str\)\b" {} \; | while read file; do
    if [[ ! "$file" =~ $BACKUP_DIR ]]; then
        echo "Found deprecated NumPy usage in: $file"
        fix_file "$file"
    fi
done

echo ""
echo "=== NumPy compatibility fixes completed! ==="
echo "Backup files saved in: $FAIRSEQ_DIR/$BACKUP_DIR"
echo ""
echo "Testing the fix..."

# Test the import
cd /home/s2587130/AVSL
python -c "
import sys
sys.path.insert(0, '/home/s2587130/AVSL/whisper_flamingo/av_hubert')
try:
    from fairseq import checkpoint_utils, utils
    print('✓ Fairseq import successful after NumPy fix!')
except Exception as e:
    print(f'✗ Import still failing: {e}')
    exit(1)
"

echo "✓ All fixes applied successfully!" 