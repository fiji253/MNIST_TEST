from pathlib import Path

class Box:
    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __init__(self):
        pass

class AnnotationSample:
    image_path: Path
    width: int
    height: int
    boxes: list[Box]