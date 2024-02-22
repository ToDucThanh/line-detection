import json
import os
import pickle
from typing import (
    List,
    Tuple,
)

import numpy as np
from tqdm import tqdm

from config import cfg


class DataPipeline:
    def __init__(self):
        self.raw_data_path = cfg.raw_data_path
        self.raw_pickle_data_path = cfg.raw_pickle_data_path
        self.processed_data_path = cfg.processed_data_path
        os.makedirs(self.processed_data_path, exist_ok=True)

    def _parse_data_from_pickle_file(self, file_name: str) -> Tuple[np.ndarray, dict]:
        pickle_file_path = os.path.join(self.raw_pickle_data_path, file_name + ".pkl")
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

    def _parse_data_from_txt_file(self, file_name: str) -> Tuple[np.ndarray, dict]:
        pickle_file_path = os.path.join(self.raw_pickle_data_path, file_name + ".pkl")
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
    def _generate_file_names(raw_data_path) -> Tuple[List, List]:
        with open(os.path.join(raw_data_path, "train.txt")) as handle:
            train_lst = [f.rstrip(".jpg\n") for f in handle.readlines()]

        with open(os.path.join(raw_data_path, "test.txt")) as handle:
            test_lst = [f.rstrip(".jpg\n") for f in handle.readlines()]
        return train_lst, test_lst

    def transform_data_from_pickle_to_json(self) -> Tuple[str, str]:
        """Return the path of train.json and test.json from pickle files"""
        train_annotations = []
        test_annotations = []

        train_lst, test_lst = self._generate_file_names(self.raw_data_path)

        for filename in tqdm(train_lst):
            _, data = self._parse_data_from_pickle_file(filename)
            train_annotations += [data]
        train_json_file_path = os.path.abspath(
            os.path.join(self.processed_data_path, "train.json")
        )
        with open(train_json_file_path, "w") as json_file:
            json.dump(train_annotations, json_file)

        for filename in tqdm(test_lst):
            _, data = self._parse_data_from_pickle_file(filename)
            test_annotations += [data]
        test_json_file_path = os.path.abspath(
            os.path.join(self.processed_data_path, "test.json")
        )
        with open(test_json_file_path, "w") as json_file:
            json.dump(test_annotations, json_file)

        return train_json_file_path, test_json_file_path

    def transform_data_from_txt_to_json(self, label_dir: str):
        """Return the path of train.json and test.json from text files\n
        `Data structure`:

        |── src/
            └── data/
                └── raw/
                    └── `train`/
                        └── images/
                            ├── 0000001.png\n
                            └── 0000002.png
                        └── labels/
                            ├── 0000001.txt\n
                            └── 0000002.txt
                    └── `tests`/
                        └── images/
                            ├── 0000003.png\n
                            └── 0000004.png
                        └── labels/
                            ├── 0000003.txt\n
                            └── 0000004.txt

        Each line of the txt file is the coordinates of one line in the corresponding image.\n
        `Example`:
            File 0000001.txt:\n
            372.43095088, 118.95949936, 374.10025597, 212.82363129\n
            435.50227356, 123.33520508, 505.86045074,  -7.40013885\n
            ....
        `Interpret`: (x1, y1, x2, y2)
        """
        files = os.listdir(label_dir)
        for file in tqdm(files):
            if file.endswith(".txt"):
                pass  # On going
