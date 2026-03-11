"""
Post-processing OCR outputs using a local LLM.
==============================================

Provides a configurable class to load a local text-generation model
in 4-bit precision to correct OCR spelling and formatting errors.
"""

from __future__ import annotations

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Default model is Qwen2.5 7B, which works well for multilingual tasks.
DEFAULT_LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert paleographer and linguist specializing in historical Spanish manuscripts. "
    "Your task is to correct OCR errors from a transcription model. Fix obvious spelling mistakes, "
    "character hallucinations, and formatting errors. "
    "CRITICAL INSTUCTIONS:\n"
    "1. Preserve the original historical language, including archaic spellings (e.g., 'ç', 'x' for 'j'), abbreviations, and missing accents.\n"
    "2. Do NOT modernize the language.\n"
    "3. Output ONLY the corrected text. Do NOT include any preamble like 'Here is the text' or explanations."
)

class LLMCorrector:
    """Loads a local LLM and uses it to correct raw OCR output."""

    def __init__(self, model_id: str = DEFAULT_LLM_MODEL_ID) -> None:
        """
        Initializes the LLM Corrector.
        
        Loads the model in 4-bit quantization to minimize VRAM usage, allowing
        it to run alongside the main Vision-Language Model.
        """
        logger.info("Initializing LLM Corrector with model: %s", model_id)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        logger.info("LLM Corrector loaded successfully.")

    def correct(self, raw_text: str) -> str:
        """
        Passes the raw text through the LLM for correction.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please correct the following transcription:\n\n{raw_text}"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Max new tokens constrained to slightly more than the input, as text length shouldn't change drastically.
        estimated_length = len(self.tokenizer.encode(raw_text))
        max_new_tokens = min(2048, int(estimated_length * 1.5) + 50)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Use greedy decoding for reproducibility
                repetition_penalty=1.05
            )
            
            # Slice the generated_ids to only get the actual generated text, omitting the prompt
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return response.strip()
