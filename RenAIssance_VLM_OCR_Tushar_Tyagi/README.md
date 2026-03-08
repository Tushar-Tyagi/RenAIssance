# RenAIssance VLM OCR Module

This module provides a modular evaluation pipeline for evaluating the zero-shot Handwritten Text Recognition (HTR) accuracy of Vision-Language Models (such as Qwen2-VL-7B-Instruct) on historical manuscripts. 

## Structure

The project is broken into several specialized modules:

- `main.py`: The main orchestrator that runs the evaluation pipeline via a clean command-line interface.
- `data_loader.py`: A utility to safely read matched pairs of images and ground-truth transcriptions from `data/test`.
- `normalize.py`: A paleographic text normalization pipeline that applies specific historical rules to standardize both the model predictions and ground-truth text before comparison. Relies on `pyspellchecker` with a comprehensive Spanish dictionary.
- `infer.py` & `vlm_models/`: Utilities to load the selected Vision-Language Model efficiently using 4-bit NF4 quantization, and dispatches the zero-shot transcription prompt.
- `evaluate.py`: Calculates Character Error Rate (CER) and Word Error Rate (WER) using `jiwer`.

## Installation

Ensure you have a GPU-enabled environment. Install the pinned dependencies provided in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Setup / Authentication
**NOTE:** Llama 3.2 Vision and MiniCPM-V 2.6 are explicitly gated on Hugging Face.
Please make sure you have run `huggingface-cli login` in your terminal and have been granted access to these models by their authors on the Hugging Face website.

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
- `--output-file <path>`: Path to save the evaluation metrics as a JSON file. If not provided, saves to `outputs/` with an auto-generated name based on model and timestamp.
- `--prompt-file <path>`: Optional parameter to override the default prompt with a plain text file.

### Supported Models

The following Vision-Language Models are supported. Specify the model using the `--model-id` argument with the corresponding HuggingFace model identifier:

- **Qwen2-VL**: `Qwen/Qwen2-VL-7B-Instruct` (default)
- **GOT-OCR**: `stepfun-ai/GOT-OCR2_0`
- **Florence-2**: `microsoft/Florence-2-base`
- **InternVL**: `OpenGVLab/InternVL2-8B`
- **MiniCPM-V**: `openbmb/MiniCPM-V-2_6`
- **Llama Vision**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Phi-3.5 Vision**: `microsoft/Phi-3.5-vision-instruct`
- **olmOCR**: `allenai/olmOCR-7B-0225-preview`

Note: Models are automatically cached locally in the `models/` directory after the first download to avoid re-downloads on subsequent runs.

## Normalization Pipeline

To assure an accurate CER and WER reading against 19th-century Spanish notarial documents, this evaluation corrects for historically divergent spellings in the evaluation pipeline before computing error scores:

1. **Cedilla Replacement**: Replaces archaic `ç` and `Ç` to `z` and `Z`.
2. **Accent Stripping**: NFD decomposes and strips all accents, while carefully keeping structural nuances like `ñ` and `Ñ`.
3. **Macron Replacements**: Normalises shorthand horizontal caps over letters (converting `q` with tilde into `que`, `ā` into `an`).
4. **Dictionary Lookups**: Automatically detects valid permutations of ambiguous `u`/`v` and `f`/`s` (e.g. `uilla` vs `villa`), preserving formatting and punctuation using `pyspellchecker`.
5. **Lower-casing**: Standardizes outputs for final jiwer calculation.
