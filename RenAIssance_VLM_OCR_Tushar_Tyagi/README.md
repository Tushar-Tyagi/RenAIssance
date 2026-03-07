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

## Usage

You can run the full evaluation pipeline using the `main.py` entrypoint:

```bash
python main.py
```

### CLI Arguments

The orchestrator supports several configuration overrides:

- `--data-dir <path>`: Root directory containing `test/images/` and `test/transcription/`. (default: `data`)
- `--model-id <name>`: The HuggingFace model space identifier (default: `Qwen/Qwen2-VL-7B-Instruct`).
- `--output-file <path>`: Optional path to save the final evaluation metrics as a JSON file.
- `--prompt-file <path>`: Optional parameter to override the default prompt with a plain text file.

## Normalization Pipeline

To assure an accurate CER and WER reading against 19th-century Spanish notarial documents, this evaluation corrects for historically divergent spellings in the evaluation pipeline before computing error scores:

1. **Cedilla Replacement**: Replaces archaic `ç` and `Ç` to `z` and `Z`.
2. **Accent Stripping**: NFD decomposes and strips all accents, while carefully keeping structural nuances like `ñ` and `Ñ`.
3. **Macron Replacements**: Normalises shorthand horizontal caps over letters (converting `q` with tilde into `que`, `ā` into `an`).
4. **Dictionary Lookups**: Automatically detects valid permutations of ambiguous `u`/`v` and `f`/`s` (e.g. `uilla` vs `villa`), preserving formatting and punctuation using `pyspellchecker`.
5. **Lower-casing**: Standardizes outputs for final jiwer calculation.
