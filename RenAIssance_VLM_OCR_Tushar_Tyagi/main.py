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
from pathlib import Path

from data_loader import load_test_pairs
from evaluate import compute_metrics, print_results, save_results
from infer import DEFAULT_MODEL_ID, DEFAULT_PROMPT, load_model, transcribe_image
from normalize import normalize

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
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file containing the zero-shot prompt.",
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

    # Generate default output file if not provided
    if args.output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_id.replace("/", "_").replace("-", "_")
        args.output_file = Path(f"outputs/eval_{model_short}_{timestamp}.json")

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Data ingestion -------------------------------------------------
    logger.info("Loading test data from '%s' …", args.data_dir)
    pairs: list[tuple[Path, str]] = load_test_pairs(data_dir=args.data_dir)

    if not pairs:
        logger.error("No test pairs found — aborting.")
        return

    logger.info("Found %d test pair(s).", len(pairs))

    # 2. Model loading --------------------------------------------------
    model = load_model(model_id=args.model_id)

    # 3 & 4. Inference + normalisation ----------------------------------
    predictions_norm: list[str] = []
    references_norm: list[str] = []

    prompt: str = DEFAULT_PROMPT
    if args.prompt_file is not None:
        logger.info("Loading prompt from '%s' …", args.prompt_file)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    for idx, (image_path, ground_truth) in enumerate(pairs, start=1):
        logger.info(
            "[%d/%d] Transcribing '%s' …", idx, len(pairs), image_path.name
        )

        raw_prediction: str = transcribe_image(
            model, image_path, prompt=prompt
        )
        logger.info("  Raw prediction (first 120 chars): %.120s", raw_prediction)

        # Normalise both sides before comparison
        pred_norm: str = normalize(raw_prediction)
        gt_norm: str = normalize(ground_truth)

        predictions_norm.append(pred_norm)
        references_norm.append(gt_norm)

    # 5. Evaluation -----------------------------------------------------
    metrics: dict[str, float] = compute_metrics(predictions_norm, references_norm)
    print_results(metrics)

    save_results(
        metrics, 
        args.output_file, 
        model_id=args.model_id, 
        prompt=prompt,
        data_dir=args.data_dir
    )
    logger.info("Saved evaluation metrics to '%s'.", args.output_file)


if __name__ == "__main__":
    main()
