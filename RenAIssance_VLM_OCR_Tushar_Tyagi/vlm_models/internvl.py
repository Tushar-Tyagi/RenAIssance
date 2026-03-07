import logging
from pathlib import Path

import torch
from PIL import Image
try:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModel

from .base import BaseVLM

logger = logging.getLogger(__name__)

class InternVLVLM(BaseVLM):
    """Wrapper for InternVL2-8B model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading InternVL model '%s'...", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
        ).eval()
        
    def transcribe(self, image_path: Path, prompt: str) -> str:
        # InternVL typically uses a generic pixel values loading technique
        try:
            pixel_values = self.model.load_image(str(image_path)).to(self.model.device).to(torch.float16)
        except AttributeError:
            # Fallback if load_image isn't exposed directly
            pixel_values = None
            logger.warning("model.load_image isn't available, make sure custom transforms are passed if needed.")
            
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        try:
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
            return str(response)
        except Exception as e:
            logger.error("InternVL chat failed: %s", e)
            return ""
