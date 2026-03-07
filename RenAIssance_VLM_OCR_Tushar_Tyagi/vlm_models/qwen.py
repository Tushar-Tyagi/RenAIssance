"""Qwen-specific Vision-Language Model implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from .base import BaseVLM

logger: logging.Logger = logging.getLogger(__name__)


class QwenVLM(BaseVLM):
    """Wrapper for Qwen2-VL specific models with 4-bit quantisation."""

    def __init__(self, model_id: str) -> None:
        """Initialize the Qwen VLM.

        Args:
            model_id: Hugging Face model identifier or local path.
        """
        logger.info("Loading Qwen model '%s' with 4-bit quantisation …", model_id)

        # Create local cache directory
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{model_id.replace('/', '_').replace('-', '_')}"

        quant_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        if os.path.exists(local_path):
            logger.info("Loading from local cache: %s", local_path)
            self.processor: AutoProcessor = AutoProcessor.from_pretrained(local_path)
            self.model: Qwen2VLForConditionalGeneration = (
                Qwen2VLForConditionalGeneration.from_pretrained(
                    local_path,
                    quantization_config=quant_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            )
        else:
            logger.info("Downloading and caching model: %s", model_id)
            self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)
            self.model: Qwen2VLForConditionalGeneration = (
                Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            )
            # Save to local cache
            self.model.save_pretrained(local_path)
            self.processor.save_pretrained(local_path)
            logger.info("Saved model to local cache: %s", local_path)

        logger.info("Qwen VLM loaded successfully.")

    def transcribe(self, image_path: Path, prompt: str) -> str:
        """Run transcription on a single image.

        Args:
            image_path: Absolute or relative path to an image.
            prompt: Text instruction for the VLM.

        Returns:
            The model's raw transcription string.
        """
        # Build the multi-modal chat message --------------------------------
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs using Qwen's vision utility ------------------------
        text_input: str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs: dict[str, Any] = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate ----------------------------------------------------------
        with torch.inference_mode():
            output_ids: torch.Tensor = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # Decode only the newly generated tokens ----------------------------
        generated_ids: list[torch.Tensor] = [
            output_ids[i][len(inputs["input_ids"][i]) :]
            for i in range(len(output_ids))
        ]

        transcription: str = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return transcription
