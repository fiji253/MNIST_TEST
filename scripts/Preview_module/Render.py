import cv2

from core_format import AnnotationSample


class PreviewRenderer:

    def draw_sample(self, sample: AnnotationSample) -> bool:
        image = cv2.imread(str(sample.image_path))

        if image is None:
            print(f"[WARN] cannot open image: {sample.image_path}")
            return True

        for box in sample.boxes:
            xmin = int(box.xmin)
            ymin = int(box.ymin)
            xmax = int(box.xmax)
            ymax = int(box.ymax)

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                color=(0, 255, 0),
                thickness=2
            )

            cv2.putText(
                image,
                box.class_name,
                (xmin, max(ymin - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        window_title = f"{sample.image_path.name} | boxes: {len(sample.boxes)}"

        cv2.imshow(window_title, image)

        key = cv2.waitKey(0)

        cv2.destroyWindow(window_title)

        if key == ord("q"):
            return False

        return True
    
    def draw_rect(self, samples: list [AnnotationSample], span: int, max_span: int):
        if span <= 0:
            raise ValueError("every must be greater than 0")
        samples_list = 0

        for index, sample in enumerate(samples, start=1):
            if index % span != 0:
                continue
            drawn_image = self.draw_sample(sample)
            samples_list.append(drawn_image)
            samples_list += 1

            if samples_list >= max_span:
                break

        cv2.destroyAllWindows()


#    def draw_every_n(
      #  self,
     #   samples: list[AnnotationSample],
    #    every: int = 10,
   #     max_previews: int = 30
  #  ):
 #       shown = 0
#
   #     for index, sample in enumerate(samples, start=1):
  #          if index % every != 0:
 #               continue
#
  #          should_continue = self.draw_sample(sample)
 #           shown += 1
#
  #          if not should_continue:
 #               break
#
 #           if shown >= max_previews:
#                break

        #cv2.destroyAllWindows()