from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def download_with_kagglehub(kaggle_slug: str, output_dir: str | Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Dataset download requires KaggleHub. Run `pip install -e .[train]`."
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(kaggle_slug))
    LOGGER.info("Kaggle dataset available at %s", dataset_path)

    events = next(dataset_path.rglob("events.csv"), None)
    if events is None:
        raise FileNotFoundError(f"Could not find events.csv under {dataset_path}")

    destination = output / "events.csv"
    if not destination.exists():
        shutil.copy2(events, destination)
    return destination
