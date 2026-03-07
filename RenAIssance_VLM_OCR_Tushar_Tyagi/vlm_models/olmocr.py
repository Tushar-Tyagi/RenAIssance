import logging
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base import BaseVLM

logger = logging.getLogger(__name__)

class OlmOCRVLM(BaseVLM):
    """Wrapper for olmOCR-7B model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading olmOCR model '%s'...", model_id)
        
        # Create local cache directory
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{model_id.replace('/', '_').replace('-', '_')}"
        
        if os.path.exists(local_path):
            logger.info("Loading from local cache: %s", local_path)
            self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
            )
        else:
            logger.info("Downloading and caching model: %s", model_id)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
            )
            # Save to local cache
            self.model.save_pretrained(local_path)
            self.processor.save_pretrained(local_path)
            logger.info("Saved model to local cache: %s", local_path)
        
        self.model.eval()
        
    def transcribe(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        text_input = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(image, text_input, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        
        input_len = inputs["input_ids"].shape[1]
        res = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
        return res.strip()
