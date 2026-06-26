import argparse

import Parsers
from parser_registry import ParserRegistry
from render import PreviewRenderer
from dataset_resolver import DatasetResolver
from pathlib import Path

# можна імпорт зробити потім з ямла

def main():

    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATASET_PATH = "train"                              # шлях до датасету в корені проекту
    FINAL_PATH = PROJ_ROOT.joinpath(DATASET_PATH)
                                                        # краще додати ще вибір формату сюди
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--format",
        required=True,
        help="Annotation format"
    )

#    parser.add_argument(
#        "--dataset",
#        required=True,
#        help="Dataset name from registry or path to dataset root"                  (на випадок консольного вводу)
#    )

#    parser.add_argument(
#        "--classes",
#        required=False,
#        default=None,
#        help="Path to classes.txt"
#    )                                                                                 (на випадок необхіідності класів)

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

    resolver = DatasetResolver()

    dataset = resolver.resolve(
        dataset=FINAL_PATH,                                                     #args.dataset, (на вмпадок консольного вводу)
        format_name=args.format
    )

    annotation_parser = ParserRegistry.create(
    dataset.format,
    images_dir=dataset.images_dir,
    labels_dir=dataset.labels_dir,
    classes_path=dataset.classes_path
)

    samples = annotation_parser.parse()

    print(f"[INFO] parsed samples: {len(samples)}")

    renderer = PreviewRenderer()
    renderer.draw_rect(
        samples=samples,
        span=args.every,
        max_span=args.max
    )


if __name__ == "__main__":
    main()