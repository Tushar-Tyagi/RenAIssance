import logging
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base import BaseVLM

logger = logging.getLogger(__name__)

class FlorenceVLM(BaseVLM):
    """Wrapper for Florence-2 model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading Florence model '%s'...", model_id)
        
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
        task_prompt = "<OCR>"
        if prompt and prompt.strip() != "":
            # Note: Florence-2 typically expects specific task tokens.
            # We use <OCR> by default or fallback to the provided prompt.
            task_prompt = prompt if "<OCR>" not in prompt else "<OCR>"
            
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.model.device, torch.float16)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        try:
            parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
            if task_prompt in parsed_answer:
                return str(parsed_answer[task_prompt])
        except Exception:
            pass
            
        return generated_text.replace(task_prompt, "").strip()
