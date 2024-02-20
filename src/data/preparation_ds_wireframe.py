import json
import os
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

pth = Path(__file__).resolve()
project_root = pth.parent.parent.parent
data_root = pth.parent

sys.path.append(str(project_root))

from data_cfg import (  # noqa
    Box,
    data_box,
)

raw_data_path = data_root / "raw"
processed_data_path = data_root / "processed"
os.makedirs(processed_data_path, exist_ok=True)

with open(os.path.join(raw_data_path, "train.txt")) as handle:
    train_lst = [f.rstrip(".jpg\n") for f in handle.readlines()]

with open(os.path.join(raw_data_path, "test.txt")) as handle:
    test_lst = [f.rstrip(".jpg\n") for f in handle.readlines()]


class WireFramePreparation:
    def __init__(self, cfg: Box = data_box):
        self.cfg = cfg

    @staticmethod
    def _parse_data_from_pickle_file(file_name: str):
        pickle_file_path = os.path.join(raw_data_path, "pointlines", file_name + ".pkl")
        with open(pickle_file_path, "rb") as fp:
            d = pickle.load(fp)
            points = d["points"]
            lines = d["lines"]
            lsgs = np.array(
                [
                    [points[i][0], points[i][1], points[j][0], points[j][1]]
                    for i, j in lines
                ],
                dtype=np.float32,
            )
            image = d["img"]
            return image, {
                "filename": file_name + ".png",
                "lines": lsgs.tolist(),
                "height": image.shape[0],
                "width": image.shape[1],
            }

    @staticmethod
    def process_data_from_pickle_to_json(train_lst: List[str], test_lst: List[str]):
        train_annotations = []
        test_annotations = []

        for filename in tqdm(train_lst):
            _, data = WireFramePreparation._parse_data_from_pickle_file(filename)
            train_annotations += [data]
        with open(os.path.join(processed_data_path, "train.json"), "w") as json_file:
            json.dump(train_annotations, json_file)

        for filename in tqdm(test_lst):
            _, data = WireFramePreparation._parse_data_from_pickle_file(filename)
            test_annotations += [data]
        with open(os.path.join(processed_data_path, "test.json"), "w") as json_file:
            json.dump(test_annotations, json_file)


if __name__ == "__main__":
    wireframe_preparation = WireFramePreparation.process_data_from_pickle_to_json(
        train_lst=train_lst, test_lst=test_lst
    )
