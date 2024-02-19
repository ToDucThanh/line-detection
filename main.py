from pathlib import Path

import cv2

from src.utils.placeholder import Box

config = Box()

# print(type(config.from_yaml))
# config.update(config.from_yaml(filename="src/models/model_cfg/base_mlsd.yaml"))
# config.update(
#     config.from_yaml(
#         filename="src/models/model_cfg/mobilev2_mlsd_tiny_512_base2_bsize24.yaml"
#     )
# )
# print(type(config))

config.update(config.from_pickle(filename="src/data/pointlines/00030043.pkl"))

image_name = config.imgname
image_path = Path("src/data/v1.1") / "train" / image_name
save_image_path = Path("src/workdir/experiments") / image_name
points = config.points
lines = config.lines

img = cv2.imread(str(image_path))
for idx, (i, j) in enumerate(lines, start=0):
    if idx == 3:
        print("Processing:", lines)
        x1, y1 = points[i]
        x2, y2 = points[j]
        print("x1, y1, x2, y2:", x1, y1, x2, y2)
        cv2.line(
            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, cv2.LINE_8
        )
        break

cv2.imwrite(str(save_image_path), img)
