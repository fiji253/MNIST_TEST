from pathlib import Path
from dataset_layout import DatasetLayout


DATASETS_ROOT = Path("datasets")


DATASET_REGISTRY = {
    "od_mvd": DatasetLayout(
        name="od_mvd",
        format="yolo",
        root_dir=DATASETS_ROOT / "od_mvd",
        images_dir=DATASETS_ROOT / "od_mvd" / "images",
        labels_dir=DATASETS_ROOT / "od_mvd" / "labels",
        classes_path=DATASETS_ROOT / "od_mvd" / "classes.txt",
    ),
}