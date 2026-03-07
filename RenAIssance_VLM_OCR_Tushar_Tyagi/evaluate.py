"""
Evaluation Module
=================

Computes Character Error Rate (CER) and Word Error Rate (WER) between
normalised predictions and normalised ground-truth strings using the
`jiwer <https://github.com/jitsi/jiwer>`_ library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jiwer import cer, wer


def compute_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute corpus-level CER and WER.

    Both ``predictions`` and ``references`` must already be normalised via the
    `normalize` module before being passed in.

    Args:
        predictions: Model predictions (one string per image).
        references:  Ground-truth transcriptions (one string per image).

    Returns:
        Dictionary with keys ``"cer"`` and ``"wer"``, each mapping to the
        corresponding float error rate (0.0 = perfect, 1.0 = 100 % error).

    Raises:
        ValueError: If the two lists differ in length or are empty.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references."
        )
    if not predictions:
        raise ValueError("Cannot compute metrics on empty input.")

    return {
        "cer": float(cer(references, predictions)),
        "wer": float(wer(references, predictions)),
    }


def print_results(metrics: dict[str, float]) -> None:
    """Pretty-print CER and WER to the console.

    Args:
        metrics: Dictionary returned by :func:`compute_metrics`.
    """
    print("\n" + "=" * 50)
    print("  HTR Evaluation Results")
    print("=" * 50)
    print(f"  Character Error Rate (CER) : {metrics['cer']:.4f}  ({metrics['cer'] * 100:.2f}%)")
    print(f"  Word Error Rate     (WER)  : {metrics['wer']:.4f}  ({metrics['wer'] * 100:.2f}%)")
    print("=" * 50 + "\n")


def save_results(
    metrics: dict[str, float], 
    output_path: str | Path,
    model_id: str | None = None,
    prompt: str | None = None,
    data_dir: str | Path | None = None
) -> None:
    """Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary returned by :func:`compute_metrics`.
        output_path: Path to the output JSON file.
        model_id: Optional identifier of the model used.
        prompt: Optional prompt text used.
        data_dir: Optional identifier of the root data directory.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {**metrics}
    if model_id is not None:
        data["model_id"] = model_id
    if prompt is not None:
        data["prompt"] = prompt

    if data_dir is not None:
        data["data_dir"] = str(data_dir)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
