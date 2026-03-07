import logging
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseVLM

logger = logging.getLogger(__name__)

class MiniCPMVLM(BaseVLM):
    """Wrapper for MiniCPM-V 2.6 model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading MiniCPM model '%s'...", model_id)
        
        # Create local cache directory
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{model_id.replace('/', '_').replace('-', '_')}"
        
        if os.path.exists(local_path):
            logger.info("Loading from local cache: %s", local_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                local_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
            )
        else:
            logger.info("Downloading and caching model: %s", model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
            )
            # Save to local cache
            self.model.save_pretrained(local_path)
            self.tokenizer.save_pretrained(local_path)
            logger.info("Saved model to local cache: %s", local_path)
        
        self.model.eval()
        
    def transcribe(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            sampling=False # greedy
        )
        return str(res)
