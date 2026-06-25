from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetLayout:
    name: str  # прибрати можна
    format: str
    root_dir: Path
    images_dir: Path
    labels_dir: Path
    classes_path: Path | None = None