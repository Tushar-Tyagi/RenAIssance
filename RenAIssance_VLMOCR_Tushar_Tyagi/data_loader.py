"""
Data Ingestion Module
=====================

Reads test images and their corresponding ground-truth transcription files
from the local directory layout::

    data/
    └── test/
        ├── images/        <- .jpg / .png files
        └── transcription/ <- .txt ground-truth files (same name as image)

"""

from __future__ import annotations

import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

#: Image file extensions that are recognised by the loader.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def load_test_pairs(
    data_dir: Path = Path("data"),
) -> list[tuple[Path, str]]:
    """Load paired (image_path, ground_truth_text) tuples from the test split.

    The function scans ``data_dir / test / images`` for image files and attempts
    to find a matching ``.txt`` file in ``data_dir / test / transcription`` whose
    name is identical.

    Images without a matching transcription are logged as warnings and skipped.

    Args:
        data_dir: Root data directory.  Defaults to ``./data``.

    Returns:
        A sorted list of ``(image_path, ground_truth_text)`` tuples.

    Raises:
        FileNotFoundError: If the images or transcription directories do not
            exist.
    """
    images_dir: Path = data_dir / "test" / "images"
    transcription_dir: Path = data_dir / "test" / "transcription"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not transcription_dir.is_dir():
        raise FileNotFoundError(
            f"Transcription directory not found: {transcription_dir}"
        )

    pairs: list[tuple[Path, str]] = []

    # Gather image files and sort by name
    image_files: list[Path] = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.warning("No image files found in %s", images_dir)
        return pairs

    for image_path in image_files:
        txt_path: Path = transcription_dir / f"{image_path.stem}.txt"

        if not txt_path.is_file():
            logger.warning(
                "No matching transcription for image '%s' — skipping.",
                image_path.name,
            )
            continue

        ground_truth: str = txt_path.read_text(encoding="utf-8").strip()
        pairs.append((image_path, ground_truth))

    logger.info(
        "Loaded %d test pairs from %s (%d images skipped due to missing GT).",
        len(pairs),
        data_dir,
        len(image_files) - len(pairs),
    )

    return pairs
