import argparse

import Parsers
from parser_registry import ParserRegistry
from Render import PreviewRenderer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--format",
        required=True,
        choices=["yolo"],
        help="Annotation format"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Path to images directory"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels directory"
    )

    parser.add_argument(
        "--classes",
        required=False,
        default=None,
        help="Path to classes.txt"
    )

    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Show every N-th image"
    )

    parser.add_argument(
        "--max",
        type=int,
        default=30,
        help="Maximum number of previews"
    )

    args = parser.parse_args()

    annotation_parser = ParserRegistry.create(
    dataset.format,
    images_dir=dataset.images_dir,
    labels_dir=dataset.labels_dir,
    classes_path=dataset.classes_path
)

    samples = annotation_parser.parse()

    print(f"[INFO] parsed samples: {len(samples)}")

    renderer = PreviewRenderer()
    renderer.draw_every_n(
        samples=samples,
        every=args.every,
        max_previews=args.max
    )


if __name__ == "__main__":
    main()