from abc import ABC
from parser_registry import AbstractParser
from pathlib import Path
from PIL import Image
from core_format import Box, AnnotationSample 

class YOLO_Parser(AbstractParser):
    def __init__(self, images_dir: str, labels_dir: str, classes_path: str | None = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes_path = Path(classes_path) if classes_path else None
        self.classes = self._load_classes()

    def parse(self) -> list[AnnotationSample]:
        image_paths = self._get_image_paths() 

        samples = []

        for image_path in image_paths:
            image = Image.open(image_path)
            #.convert("RGB")
            width, height = image.size

            label_path = self.labels_dir / f"{image_path.stem}.txt" # ?

            boxes = self._read_label_file(
                label_path=label_path,
                image_width=width,
                image_height=height
            )

            sample = AnnotationSample(
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes
            )

            samples.append(sample)

        return samples

    def _load_classes(self) -> list[str]:
        if self.classes_path is None:
            return []

        if not self.classes_path.exists():
            print(f"[WARN] classes file not found: {self.classes_path}")
            return []

        with open(self.classes_path, "r", encoding="utf-8") as file:
            return [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

    def _get_image_paths(self) -> list[Path]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        return sorted([
            path for path in self.images_dir.iterdir()
            if path.suffix.lower() in image_extensions
        ])

    def _read_label_file(
        self,
        label_path: Path,
        image_width: int,
        image_height: int
    ) -> list[Box]:
        boxes = []

        if not label_path.exists():
            print(f"[WARN] label file not found: {label_path}")
            return boxes

        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"[WARN] bad YOLO annotation line: "
                    f"{label_path}, line {line_number}: {line}"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
            except ValueError:
                print(
                    f"[WARN] cannot parse YOLO annotation line: "
                    f"{label_path}, line {line_number}: {line}"
                )
                continue

            xmin, ymin, xmax, ymax = self._yolo_to_xyxy(
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height
            )

            class_name = self._get_class_name(class_id)

            boxes.append(Box(
                class_id=class_id,
                class_name=class_name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            ))

        return boxes

    def _get_class_name(self, class_id: int) -> str:
        if self.classes and 0 <= class_id < len(self.classes):
            return self.classes[class_id]

        return f"class_{class_id}"

    @staticmethod
    def _yolo_to_xyxy(
        x_center: float,
        y_center: float,
        box_width: float,
        box_height: float,
        image_width: int,
        image_height: int
    ) -> tuple[float, float, float, float]:
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        box_width_px = box_width * image_width
        box_height_px = box_height * image_height

        xmin = x_center_px - box_width_px / 2
        ymin = y_center_px - box_height_px / 2
        xmax = x_center_px + box_width_px / 2
        ymax = y_center_px + box_height_px / 2

        return xmin, ymin, xmax, ymax