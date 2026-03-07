#!/bin/bash

# Script to test all available VLM models on both data and data_alltest folders
# Results are automatically saved to outputs/ with auto-generated filenames
# All output is logged to outputs/test_all_log.txt

# Redirect all output to log file
exec > >(tee outputs/test_all_log.txt) 2>&1

# Ensure outputs directory exists
mkdir -p outputs

echo "Starting comprehensive model evaluation at $(date)"
echo "=================================================="

# Array of model IDs
models=(
    # "Qwen/Qwen2-VL-7B-Instruct"
    "stepfun-ai/GOT-OCR2_0"
    "microsoft/Florence-2-base"
    "OpenGVLab/InternVL2-8B"
    "openbmb/MiniCPM-V-2_6"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "microsoft/Phi-3.5-vision-instruct"
    "M4-ai/olmOCR-7B-0225-preview"
)

# Array of data directories
data_dirs=(
    "data"
    "data_alltest"
)

# Loop through each model
for model in "${models[@]}"; do
    echo "Testing model: $model"
    
    # Loop through each data directory
    for data_dir in "${data_dirs[@]}"; do
        echo "  Using data directory: $data_dir"
        
        # Run the evaluation
        python main.py --model-id "$model" --data-dir "$data_dir"
        
        echo "  Completed evaluation for $model with $data_dir"
        echo ""
    done
done

echo "All evaluations completed. Results saved in outputs/ folder."
echo "Log file: outputs/test_all_log.txt"
echo "Completed at $(date)"