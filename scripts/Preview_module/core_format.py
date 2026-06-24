from pathlib import Path
from dataclasses import dataclass

@dataclass
class Box:
    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class AnnotationSample:
    image_path: Path
    width: int
    height: int
    boxes: list[Box]