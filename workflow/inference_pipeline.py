import os
from typing import (
    List,
    Optional,
)

import cv2
import numpy as np
import torch
from torch.nn import functional as F


class InferencePipeline:
    def __init__(self, model, device, model_weight_path: str):
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=True)
        self.model.eval()

    def read_and_preprocess_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess(self, img: np.ndarray):
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def process(
        self,
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> List[float]:
        if image is not None:
            img = self.preprocess(image)
        else:
            img = self.read_and_preprocess_image(image_path)
        lines = self._pred_lines(img, self.model, [512, 512], 0.1, 20)
        return img, lines

    def postprocess(
        self, img: np.ndarray, lines: List[float], image_save_name: str, save_dir: str
    ):
        """Draw lines on the original image"""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for line in lines:
            cv2.line(
                img,
                (int(line[0]), int(line[1])),
                (int(line[2]), int(line[3])),
                (0, 200, 200),
                1,
                16,
            )
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/{image_save_name}.jpg", img)

    def run(
        self,
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        image_save_name: str = "sample_image",
        save_dir: str = "./src/workdir/experiments/output",
        is_saving: bool = True,
    ):
        img, lines = self.process(image_path=image_path, image=image)
        if is_saving:
            self.postprocess(
                img=img, lines=lines, image_save_name=image_save_name, save_dir=save_dir
            )
        return img, lines

    def _pred_lines(self, image, model, input_shape=[512, 512], score_thr=0.10, dist_thr=20.0):
        h, w, _ = image.shape
        h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

        resized_image = np.concatenate(
            [
                cv2.resize(
                    image,
                    (input_shape[0], input_shape[1]),
                    interpolation=cv2.INTER_AREA,
                ),
                np.ones([input_shape[0], input_shape[1], 1]),
            ],
            axis=-1,
        )

        resized_image = resized_image.transpose((2, 0, 1))
        batch_image = np.expand_dims(resized_image, axis=0).astype("float32")
        batch_image = (batch_image / 127.5) - 1.0

        batch_image = torch.from_numpy(batch_image).float().to(self.device)
        outputs = model(batch_image)
        pts, pts_score, vmap = InferencePipeline._deccode_output_score_and_ptss(outputs, 200, 3)
        start = vmap[:, :, :2]
        end = vmap[:, :, 2:]
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        segments_list = []
        for center, score in zip(pts, pts_score):
            y, x = center
            distance = dist_map[y, x]
            if score > score_thr and distance > dist_thr:
                disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
                x_start = x + disp_x_start
                y_start = y + disp_y_start
                x_end = x + disp_x_end
                y_end = y + disp_y_end
                segments_list.append([x_start, y_start, x_end, y_end])

        lines = 2 * np.array(segments_list)  # 256 > 512
        lines[:, 0] = lines[:, 0] * w_ratio
        lines[:, 1] = lines[:, 1] * h_ratio
        lines[:, 2] = lines[:, 2] * w_ratio
        lines[:, 3] = lines[:, 3] * h_ratio

        return lines

    @staticmethod
    def _deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
        """
        tpMap:
        center: tpMap[1, 0, :, :]
        displacement: tpMap[1, 1:5, :, :]
        """
        b, c, h, w = tpMap.shape
        assert b == 1, "only support bsize==1"
        displacement = tpMap[:, 1:5, :, :][0]
        center = tpMap[:, 0, :, :]
        heat = torch.sigmoid(center)
        hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
        keep = (hmax == heat).float()
        heat = heat * keep
        heat = heat.reshape(
            -1,
        )

        scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
        yy = torch.floor_divide(indices, w).unsqueeze(-1)
        xx = torch.fmod(indices, w).unsqueeze(-1)
        ptss = torch.cat((yy, xx), dim=-1)

        ptss = ptss.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        displacement = displacement.detach().cpu().numpy()
        displacement = displacement.transpose((1, 2, 0))

        return ptss, scores, displacement
