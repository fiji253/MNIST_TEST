from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core_format import AnnotationSample
from validator import BoxValidator


class PreviewRenderer:
    def draw_every_n(
        self,
        samples: list[AnnotationSample],
        every: int = 10,
        max_previews: int = 30
    ):
        shown = 0

        for index, sample in enumerate(samples, start=1):
            if index % every != 0:
                continue

            self.draw_sample(sample)
            shown += 1

            if shown >= max_previews:
                break

    def draw_sample(self, sample: AnnotationSample):
        image = Image.open(sample.image_path).convert("RGB")

        fig, ax = plt.subplots(1, figsize=(12, 8))

        ax.imshow(image)
        ax.set_title(
            f"{sample.image_path.name} | "
            f"{sample.width}x{sample.height} | "
            f"boxes: {len(sample.boxes)}"
        )
        ax.axis("off")

        for box in sample.boxes:
            problems = BoxValidator.validate_box(sample, box)

            edge_color = "red" if problems else "lime"
            label_bg_color = "red" if problems else "green"

            rect_width = box.xmax - box.xmin
            rect_height = box.ymax - box.ymin

            rectangle = patches.Rectangle(
                (box.xmin, box.ymin),
                rect_width,
                rect_height,
                linewidth=2,
                edgecolor=edge_color,
                facecolor="none"
            )

            ax.add_patch(rectangle)

            label_text = f"{box.class_name} ({box.class_id})"

            if problems:
                label_text += " | " + "; ".join(problems)

            ax.text(
                box.xmin,
                max(box.ymin - 5, 0),
                label_text,
                fontsize=9,
                color="white",
                bbox=dict(
                    facecolor=label_bg_color,
                    alpha=0.75
                )
            )

        plt.show()