"""
HTR Zero-Shot Evaluation — Main Orchestrator
=============================================

Usage::

    python main.py                        # defaults, saves to outputs/
    python main.py --data-dir ./data      # custom data root
    python main.py --model-id Qwen/Qwen2-VL-2B-Instruct  # different model
"""

from __future__ import annotations

import argparse
import datetime
import logging
import json
from pathlib import Path

from data_loader import load_test_pairs
from evaluate import compute_metrics, print_results, save_results
from infer import DEFAULT_MODEL_ID, DEFAULT_PROMPT, load_model, transcribe_image
from normalize import normalize
from llm_corrector import LLMCorrector, DEFAULT_LLM_MODEL_ID

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with ``data_dir`` and ``model_id`` attributes.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate zero-shot HTR accuracy of a Vision-Language Model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory containing test/images and test/transcription.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model identifier (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save the evaluation metrics as a JSON file. If not provided, saves to outputs/ with auto-generated name.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights to load.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file containing the zero-shot prompt.",
    )
    parser.add_argument(
        "--use-llm-correction",
        action="store_true",
        help="Enable local LLM-based spelling correction of the VLM output.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL_ID,
        help=f"HuggingFace model ID for the local LLM corrector (default: {DEFAULT_LLM_MODEL_ID}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full HTR evaluation pipeline.

    Steps:

    1. Load test image / ground-truth pairs.
    2. Load the quantised VLM.
    3. Transcribe each image with the VLM.
    4. Normalise both prediction and ground truth.
    5. Compute and print CER & WER.
    """
    args: argparse.Namespace = parse_args()
    # Robustness: strip trailing commas that might come from typos in shell scripts
    args.model_id = args.model_id.strip(",")

    # Generate default output file if not provided
    if args.output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_id.replace("/", "_").replace("-", "_")
        if args.adapter_path:
            adapter_name = Path(args.adapter_path).name
            args.output_file = Path(f"outputs/eval_{model_short}_{adapter_name}_{timestamp}.json")
    # Check if we should load a cache
    cache: dict[str, dict[str, str]] = {}
    if args.output_file.exists():
        logger.info("Found existing output file '%s'. Loading cached predictions...", args.output_file)
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                if "results_details" in cached_data:
                    for detail in cached_data["results_details"]:
                        if "image_file" in detail:
                            cache[detail["image_file"]] = detail
            logger.info("Loaded %d cached predictions.", len(cache))
        except Exception as e:
            logger.warning("Failed to load existing output file for caching: %s", e)

    # 1. Data ingestion -------------------------------------------------
    logger.info("Loading test data from '%s' …", args.data_dir)
    pairs: list[tuple[Path, str]] = load_test_pairs(data_dir=args.data_dir)

    if not pairs:
        logger.error("No test pairs found — aborting.")
        return

    logger.info("Found %d test pair(s).", len(pairs))

    # 2. VLM Inference --------------------------------------------------
    raw_predictions: list[str] = []
    
    prompt: str = DEFAULT_PROMPT
    if args.prompt_file is not None:
        logger.info("Loading prompt from '%s' …", args.prompt_file)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    # Only load VLM if we actually have images that need VLM processing
    needs_vlm = any(
        image_path.name not in cache or "vlm_prediction_raw" not in cache[image_path.name]
        for image_path, _ in pairs
    )

    model = None
    if needs_vlm:
        model = load_model(model_id=args.model_id, adapter_path=args.adapter_path)

    for idx, (image_path, _) in enumerate(pairs, start=1):
        image_name = image_path.name
        if image_name in cache and "vlm_prediction_raw" in cache[image_name]:
            raw_prediction = cache[image_name]["vlm_prediction_raw"]
            logger.info("[%d/%d] Skipping VLM for '%s' (found in cache).", idx, len(pairs), image_name)
        else:
            logger.info("[%d/%d] Transcribing '%s' …", idx, len(pairs), image_name)
            raw_prediction: str = transcribe_image(
                model, image_path, prompt=prompt
            )
            logger.info("  Raw prediction (first 120 chars): %.120s", raw_prediction)
            
            # Save back to cache dynamically to mark it as found for subsequent steps
            if image_name not in cache:
                cache[image_name] = {"image_file": image_name}
            cache[image_name]["vlm_prediction_raw"] = raw_prediction
            
        raw_predictions.append(raw_prediction)

    vlm_predictions = list(raw_predictions)
    llm_predictions = []

    # Free up VLM memory if we are going to load an LLM
    if args.use_llm_correction:
        if model is not None:
            logger.info("Unloading VLM to free memory for LLM Corrector...")
            import gc
            import torch
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Only load the LLM if there are actual missing LLM predictions
        needs_llm = any(
            image_path.name not in cache or "llm_prediction_raw" not in cache[image_path.name]
            for image_path, _ in pairs
        )

        if needs_llm:
            corrector = LLMCorrector(model_id=args.llm_model)
        else:
            corrector = None

        for i in range(len(raw_predictions)):
            image_name = pairs[i][0].name
            if image_name in cache and "llm_prediction_raw" in cache[image_name]:
                corrected_pred = cache[image_name]["llm_prediction_raw"]
                logger.info("  [%d/%d] Skipping LLM for '%s' (found in cache).", i + 1, len(raw_predictions), image_name)
            else:
                logger.info("  [%d/%d] Correcting text with LLM for '%s'...", i + 1, len(raw_predictions), image_name)
                original_pred = raw_predictions[i]
                if corrector is not None:
                    corrected_pred = corrector.correct(original_pred)
                else: 
                    # Fallback just in case, though logically unreachable due to needs_llm check
                    corrected_pred = original_pred 
                logger.info("  Corrected prediction (first 120 chars): %.120s", corrected_pred)
                
                if image_name not in cache:
                    cache[image_name] = {"image_file": image_name}
                cache[image_name]["llm_prediction_raw"] = corrected_pred

            llm_predictions.append(corrected_pred)
            raw_predictions[i] = corrected_pred

    # 3. Normalisation --------------------------------------------------
    predictions_norm: list[str] = []
    references_norm: list[str] = []
    results_details = []

    for i, (image_path, ground_truth) in enumerate(pairs):
        gt_norm: str = normalize(ground_truth)
        references_norm.append(gt_norm)

        pred = raw_predictions[i]
        pred_norm = normalize(pred)
        predictions_norm.append(pred_norm)

        detail = {
            "image_file": image_path.name,
            "ground_truth_raw": ground_truth,
            "ground_truth_norm": gt_norm,
            "vlm_prediction_raw": vlm_predictions[i],
            "final_prediction_raw": pred,
            "final_prediction_norm": pred_norm,
        }
        if llm_predictions:
             detail["llm_prediction_raw"] = llm_predictions[i]
             detail["llm_prediction_norm"] = pred_norm # the final_prediction is the LLM prediction when LLM is used
             
        results_details.append(detail)

    # 4. Evaluation -----------------------------------------------------
    metrics: dict[str, float] = compute_metrics(predictions_norm, references_norm)
    print_results(metrics)

    save_results(
        metrics, 
        args.output_file, 
        model_id=args.model_id, 
        prompt=prompt,
        data_dir=args.data_dir,
        adapter_path=args.adapter_path,
        results_details=results_details
    )
    logger.info("Saved evaluation metrics to '%s'.", args.output_file)


if __name__ == "__main__":
    main()
