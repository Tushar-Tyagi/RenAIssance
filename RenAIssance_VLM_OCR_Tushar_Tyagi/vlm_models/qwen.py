"""Qwen-specific Vision-Language Model implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

# Optional dynamic imports for 2.5 and 3.0+ architectures
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None

from .base import BaseVLM

logger: logging.Logger = logging.getLogger(__name__)


class QwenVLM(BaseVLM):
    """Wrapper for Qwen-VL family models (2.0, 2.5, 3.0+) with 4-bit quantisation."""

    def __init__(self, model_id: str, adapter_path: str | None = None) -> None:
        """Initialize the Qwen VLM.

        Args:
            model_id: Hugging Face model identifier or local path.
            adapter_path: Optional path to LoRA adapter weights.
        """
        logger.info("Loading Qwen model '%s' with 4-bit quantisation …", model_id)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "qwen2_vl")
        logger.info("Detected model architecture type: %s", model_type)

        quant_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)
        
        # Dispatch to the correct class based on architecture
        if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration:
            ModelClass = Qwen3VLMoeForConditionalGeneration
        elif model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration:
            ModelClass = Qwen3VLForConditionalGeneration
        elif model_type == "qwen2_5_vl" and Qwen2_5_VLForConditionalGeneration:
            ModelClass = Qwen2_5_VLForConditionalGeneration
        else:
            # Fallback to Qwen2-VL for anything else
            ModelClass = Qwen2VLForConditionalGeneration
            if model_type not in ["qwen2_vl", "got_qwen_vl"]:
                logger.warning(
                    "Model type '%s' not explicitly handled. Falling back to Qwen2VLForConditionalGeneration.",
                    model_type
                )
            
        self.model = ModelClass.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if adapter_path:
            logger.info("Loading LoRA adapter from '%s' …", adapter_path)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        logger.info("Successfully loaded %s with %s architecture.", model_id, ModelClass.__name__)

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
