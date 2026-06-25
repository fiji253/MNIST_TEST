from pathlib import Path

from dataset_layout import DatasetLayout
from dataset_registry import DATASET_REGISTRY


class DatasetResolver:
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def resolve(self, dataset: str, format_name: str | None = None) -> DatasetLayout:
        if format_name is None:
            raise ValueError("Annotation format is required")

        layout = DATASET_REGISTRY.get(dataset)

        if layout is not None:
            return layout

        root = Path(dataset)

        images_dir = root / "images"
        labels_dir = root / "labels"

        if not self._has_pairs(images_dir, labels_dir):
            images_dir, labels_dir = self._discover(root)

        return DatasetLayout(
            name=root.name,
            format=format_name,
            root_dir=root,
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes_path=self._find_classes(root)
        )

    def _discover(self, root: Path) -> tuple[Path, Path]:
        dirs = [root] + [
            path for path in root.rglob("*")
            if path.is_dir()
        ]

        for images_dir in dirs:
            for labels_dir in dirs:
                if self._has_pairs(images_dir, labels_dir):
                    return images_dir, labels_dir

        raise FileNotFoundError(f"Cannot find matching images/labels in: {root}")

    def _has_pairs(self, images_dir: Path, labels_dir: Path) -> bool:
        if not images_dir.exists() or not labels_dir.exists():
            return False

        image_names = {
            path.stem
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMG_EXT
        }

        label_names = {
            path.stem
            for path in labels_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }

        return bool(image_names & label_names)

    def _find_classes(self, root: Path) -> Path | None:
        classes_path = root / "classes.txt"

        if classes_path.exists():
            return classes_path

        return next(root.rglob("classes.txt"), None)