"""Base Vision-Language Model interface."""

from __future__ import annotations

import abc
from pathlib import Path


class BaseVLM(abc.ABC):
    """Abstract base class for all VLM wrappers."""

    @abc.abstractmethod
    def transcribe(self, image_path: Path, prompt: str) -> str:
        """Transcribe an image given a prompt.

        Args:
            image_path: Absolute or relative path to an image.
            prompt: Text prompt to guide the transcription.

        Returns:
            The raw string output from the model.
        """
        ...
