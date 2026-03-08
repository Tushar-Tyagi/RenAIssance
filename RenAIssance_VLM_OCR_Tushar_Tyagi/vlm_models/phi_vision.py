import logging
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base import BaseVLM

logger = logging.getLogger(__name__)

class PhiVisionVLM(BaseVLM):
    """Wrapper for Phi-3.5 Vision model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading Phi Vision model '%s'...", model_id)
        
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
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        text_input = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text_input, [image], return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        input_len = inputs["input_ids"].shape[1]
        res = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
        return res.strip()
