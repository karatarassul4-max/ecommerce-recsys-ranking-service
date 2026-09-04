from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    project: dict[str, Any]
    dataset: dict[str, Any]
    split: dict[str, Any]
    models: dict[str, Any]
    tracking: dict[str, Any]


def load_config(path: str | Path) -> ProjectConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig(
        project=data["project"],
        dataset=data["dataset"],
        split=data["split"],
        models=data["models"],
        tracking=data["tracking"],
    )
