#!/bin/bash

# Ensure outputs directory exists
mkdir -p outputs
LOGFILE="outputs/test_qwen_log.txt"

echo "Starting Qwen model evaluation at $(date)" > "$LOGFILE"
echo "==================================================" >> "$LOGFILE"

# Define the models and datasets to test
MODELS=(
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2-VL-32B-Instruct"
    "Qwen/Qwen3-VL-7B-Instruct"
)
DATASETS=("data" "data_alltest")

for model in "${MODELS[@]}"; do
    for data_dir in "${DATASETS[@]}"; do
        echo "Testing model: $model" | tee -a "$LOGFILE"
        echo "  Using data directory: $data_dir" | tee -a "$LOGFILE"
        python main.py --model-id "$model" --data-dir "$data_dir" 2>&1 | tee -a "$LOGFILE"
        echo "--------------------------------------------------" >> "$LOGFILE"
    done
done

echo "==================================================" >> "$LOGFILE"
echo "Evaluation complete at $(date)" >> "$LOGFILE"