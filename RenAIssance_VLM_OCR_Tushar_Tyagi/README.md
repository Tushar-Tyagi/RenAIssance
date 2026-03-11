# RenAIssance VLM OCR Module

This module provides a modular evaluation pipeline for evaluating the zero-shot Handwritten Text Recognition (HTR) accuracy of Vision-Language Models (such as Qwen2-VL-7B-Instruct) on historical manuscripts. 

## Structure

The project is broken into several specialized modules:

- `main.py`: The main orchestrator that runs the evaluation pipeline via a clean command-line interface.
- `data_loader.py`: A utility to safely read matched pairs of images and ground-truth transcriptions from `data/test`.
- `normalize.py`: A paleographic text normalization pipeline that applies specific historical rules to standardize both the model predictions and ground-truth text before comparison. Relies on `pyspellchecker` with a comprehensive Spanish dictionary.
- `infer.py` & `vlm_models/`: Utilities to load the selected Vision-Language Model efficiently using 4-bit NF4 quantization, and dispatches the zero-shot transcription prompt.
- `evaluate.py`: Calculates Character Error Rate (CER) and Word Error Rate (WER) using `jiwer`.
- `finetune.py`: Script to fine-tune Vision-Language Models on the provided datasets.
- `prepare_xml_dataset.py`: A utility script to parse PAGE XML formatted datasets, extracting the transcriptions into JSONL format compatible with `finetune.py`.

## Installation

Ensure you have a GPU-enabled environment. Install the pinned dependencies provided in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Setup

Due to its size and underlying requirements, you will need a GPU-enabled environment.

## Usage

You can run the full evaluation pipeline using the `main.py` entrypoint:

```bash
python main.py
```

To test all available models on both `data` and `data_alltest` folders, use the provided script:

```bash
./test_all.sh
```

This will run evaluations for all supported models and save results automatically to the `outputs/` folder. All console output is also logged to `outputs/test_all_log.txt` for analysis.

### CLI Arguments

The orchestrator supports several configuration overrides:

- `--data-dir <path>`: Root directory containing `test/images/` and `test/transcription/`. (default: `data`)
- `--model-id <name>`: The HuggingFace model space identifier (default: `Qwen/Qwen2-VL-7B-Instruct`).
- `--adapter-path <path>`: Optional path to load fine-tuned LoRA adapter weights (e.g. `./checkpoints/ocr_vlm_lora`).
- `--output-file <path>`: Path to save the evaluation metrics as a JSON file. If not provided, saves to `outputs/` with an auto-generated name based on model and timestamp.
- `--prompt-file <path>`: Optional parameter to override the default prompt with a plain text file.
- `--use-llm-correction`: Flag to enable a post-processing step that corrects OCR spelling/formatting errors using a local 4-bit quantized text-generation LLM.
- `--llm-model <name>`: HuggingFace model space identifier for the text-generation LLM corrector (default: `Qwen/Qwen2.5-7B-Instruct`).

### Supported Models

It natively supports passing any Qwen2-VL, Qwen2.5-VL, or Qwen3-VL series string. Standard examples include:

- **Qwen2-VL (Small/Fast)**: `Qwen/Qwen2-VL-2B-Instruct`
- **Qwen2-VL (Base)**: `Qwen/Qwen2-VL-7B-Instruct` (default)
- **Qwen2-VL (Large)**: `Qwen/Qwen2-VL-32B-Instruct`
- **Qwen2.5-VL (Base)**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Qwen2.5-VL (Small/Fast)**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Qwen3-VL (Base)**: `Qwen/Qwen3-VL-7B-Instruct`

Note: Models are automatically cached locally in the `models/` directory after the first download to avoid re-downloads on subsequent runs.

## Training & Dataset Preparation

We now provide an end-to-end framework to fine-tune modern VLMs based on custom datasets! 

### 1. Extracting Transcriptions
If your transcription dataset uses the **PAGE XML format** (commonly utilized for historical archival OCR), you can automatically parse your `.xml` files into the optimal `.jsonl` structure used by the SFTTrainer using the newly provided `prepare_xml_dataset.py` script:

```bash
python prepare_xml_dataset.py \
    --xml_dir path/to/your/xml/folder \
    --output_jsonl data/train_annotations.jsonl
```

You can optionally declare `--image_dir_prefix path/to/images/` to prepend a root path directly inside the JSON files if your image paths will change, or `--save_txt` to generate identical raw text transcriptions adjacent to each XML file.

### 2. Fine-tuning
Once you have your `train_annotations.jsonl` matching your `images/` directory, you can run the fine-tuning script with flexible command-line arguments:

```bash
# Default configuration (uses all prompts from prompts/ directory)
python finetune.py

# Custom model and data paths
python finetune.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --annotation_file "data/train_annotations.jsonl" \
    --base_image_dir "data/"

# Custom training hyperparameters
python finetune.py \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lora_r 32

# Full custom configuration with custom prompts directory
python finetune.py \
    --model "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --annotation_file "data/train_annotations.jsonl" \
    --output_dir "./my_checkpoints" \
    --prompts_dir "./my_prompts" \
    --epochs 10 \
    --batch_size 2
```

#### Available Arguments

- `--model <model_id>`: Hugging Face model identifier (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--annotation_file <path>`: Path to JSONL annotation file (default: `data/train_annotations.jsonl`)
- `--base_image_dir <path>`: Base directory for image files (default: `data/`)
- `--output_dir <path>`: Directory to save LoRA adapter weights (default: `./checkpoints/ocr_vlm_lora`)
- `--epochs <int>`: Number of training epochs (default: `3`)
- `--batch_size <int>`: Per-device training batch size (default: `2`)
- `--grad_accum_steps <int>`: Gradient accumulation steps (default: `8`)
- `--learning_rate <float>`: Learning rate (default: `2e-4`)
- `--lora_r <int>`: LoRA rank parameter (default: `16`)
- `--lora_alpha <int>`: LoRA alpha scaling parameter (default: `32`)
- `--lora_dropout <float>`: LoRA dropout probability (default: `0.05`)
- `--prompts_dir <path>`: Directory containing `.txt` prompt files for diverse training (default: `prompts`)

#### Prompt Diversity for Generalization

The fine-tuning pipeline automatically loads all `.txt` files from the `prompts/` directory and randomly samples from them during training. This helps the model generalize better by learning to respond to various phrasings of the transcription task. 

Currently available prompts:
- `default.txt`: "Transcribe the handwritten text in this image exactly as written. Do not correct spelling, punctuation, or grammar. Preserve all original characters."
- `alternative_1.txt`: "Please transcribe the text from the provided image of a historical document. Write the text exactly as it appears in the image. Keep any historical spellings, punctuation marks, or symbols. If a word is crossed out but readable, transcribe it. Do not attempt to modernise the language or fix perceived errors."

You can add more `.txt` files to the `prompts/` directory to further improve model robustness and generalization.

This efficiently targets the `q_proj`, `v_proj`, `k_proj`, and `o_proj` attention matrices with 4-bit LoRA adapters while applying optimized mixed-precision `bfloat16` and Paged AdamW optimizers to maximize fine-tuning stability on standard consumer GPU hardware.

### 3. Evaluating Fine-Tuned Models

After fine-tuning, you can pass the saved adapter path to the `main.py` evaluation pipeline to test the newly adjusted weights:

```bash
python main.py \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --adapter-path "./checkpoints/ocr_vlm_lora" \
    --data-dir "data"
```

The resulting evaluation JSON will automatically reflect the adapter's use in both its filename and internal metadata.

## Normalization Pipeline

To assure an accurate CER and WER reading against 19th-century Spanish notarial documents, this evaluation corrects for historically divergent spellings in the evaluation pipeline before computing error scores:

1. **Cedilla Replacement**: Replaces archaic `ç` and `Ç` to `z` and `Z`.
2. **Accent Stripping**: NFD decomposes and strips all accents, while carefully keeping structural nuances like `ñ` and `Ñ`.
3. **Macron Replacements**: Normalises shorthand horizontal caps over letters (converting `q` with tilde into `que`, `ā` into `an`).
4. **Dictionary Lookups**: Automatically detects valid permutations of ambiguous `u`/`v` and `f`/`s` (e.g. `uilla` vs `villa`), preserving formatting and punctuation using `pyspellchecker`.
5. **Lower-casing**: Standardizes outputs for final jiwer calculation.
