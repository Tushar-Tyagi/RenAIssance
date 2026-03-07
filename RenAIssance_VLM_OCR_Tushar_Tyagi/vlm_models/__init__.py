"""VLM implementations for OCR abstraction."""

from .base import BaseVLM
from .qwen import QwenVLM

__all__ = ["BaseVLM", "QwenVLM"]
