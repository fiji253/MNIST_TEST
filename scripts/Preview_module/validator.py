from core_format import AnnotationSample, Box


class BoxValidator:
    @staticmethod
    def validate_box(sample: AnnotationSample, box: Box) -> list[str]:
        problems = []

        if box.xmax <= box.xmin:
            problems.append("xmax <= xmin")

        if box.ymax <= box.ymin:
            problems.append("ymax <= ymin")

        if box.xmin < 0:
            problems.append("xmin < 0")

        if box.ymin < 0:
            problems.append("ymin < 0")

        if box.xmax > sample.width:
            problems.append("xmax > image width")

        if box.ymax > sample.height:
            problems.append("ymax > image height")

        return problems