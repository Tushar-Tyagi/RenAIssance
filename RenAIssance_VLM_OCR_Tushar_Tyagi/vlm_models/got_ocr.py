import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseVLM

logger = logging.getLogger(__name__)

class GotOCRVLM(BaseVLM):
    """Wrapper for GOT-OCR2.0 model."""
    
    def __init__(self, model_id: str):
        logger.info("Loading GOT-OCR model '%s'...", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
        )
        self.model.eval()
        
    def transcribe(self, image_path: Path, prompt: str) -> str:
        try:
            res = self.model.chat(self.tokenizer, str(image_path), ocr_type='ocr')
            return str(res)
        except Exception as e:
            logger.warning("Standard GOT-OCR 'ocr' mode failed, falling back: %s", e)
            res = self.model.chat(self.tokenizer, str(image_path), prompt=prompt)
            if isinstance(res, tuple):
                return str(res[0])
            return str(res)
