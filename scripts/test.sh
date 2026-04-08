#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# Single test configuration only
dataset='inria_buildings'
split='0.5'
method='heightmatch'
model='dinov2_small'

# Path settings
config="configs/${dataset}.yaml"
save_path="saved/${dataset}/${split}/${method}/${model}"
test_id_path="splits/${dataset}/test.txt"

# Create output directory
mkdir -p "$save_path"

# Print current configuration
echo "================================================"
echo "Testing:"
echo "Dataset: $dataset"
echo "Method: $method"
echo "Split: $split"
echo "Model: $model"
echo "================================================"

# Check whether required files exist
if [ ! -f "$config" ]; then
    echo "Error: Config file not found: $config"
    exit 1
fi

if [ ! -f "$test_id_path" ]; then
    echo "Error: Test ID file not found: $test_id_path"
    exit 1
fi

# Run test script and save log
python3 test.py \
    --config "$config" \
    --save-path "$save_path" \
    --test-id-path "$test_id_path" 2>&1 | tee "$save_path/test_$now.log"

echo "================================================"
echo "Completed: $dataset/$split/$method"
echo "================================================"