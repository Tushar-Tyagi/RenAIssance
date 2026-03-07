"""
VLM Inference Module
====================

Unified entrypoint to load initialized Vision-Language Models
and dispatch image transcribing logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vlm_models.base import BaseVLM
from vlm_models.qwen import QwenVLM

logger: logging.Logger = logging.getLogger(__name__)

#: Default Hugging Face model identifier.
DEFAULT_MODEL_ID: str = "Qwen/Qwen2-VL-7B-Instruct"

#: Default zero-shot prompt sent to the VLM for each image.
DEFAULT_PROMPT: str = (
    "Transcribe the handwritten text in this image exactly as written. "
    "Do not correct spelling, punctuation, or grammar. "
    "Preserve all original characters."
)


def load_model(model_id: str = DEFAULT_MODEL_ID) -> BaseVLM:
    """Load a Vision-Language Model wrapper appropriately sized for the model_id.

    Args:
        model_id: Hugging Face model identifier or local path.

    Returns:
        A VLM instance implementing :class:`BaseVLM`.
    """
    model_id_lower = model_id.lower()

    # Simple dispatcher based on model_id heuristics
    if "qwen" in model_id_lower:
        return QwenVLM(model_id)

    raise ValueError(f"No corresponding wrapper implementation for model: {model_id}")


def transcribe_image(
    model: BaseVLM,
    image_path: Path,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """Run transcription on a single image.

    Args:
        model:      The instantiated VLM instance (e.g. QwenVLM).
        image_path: Absolute or relative path to a ``.jpg`` / ``.png`` image.
        prompt:     Instruction prompt injected before image.

    Returns:
        The model's raw transcription string.
    """
    return model.transcribe(image_path=image_path, prompt=prompt)
