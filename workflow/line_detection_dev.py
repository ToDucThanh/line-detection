from typing import List

import cv2
import numpy as np
import torch

from config import cfg
from src.models.networks.mobilev2_mlsd_tiny_net import MobileV2_MLSD_Tiny
from src.utils.placeholder import Box

from .inference_pipeline import InferencePipeline


class LineDetectionDemo:
    required_args = ["video_path", "image_path"]

    def __init__(self, **kwargs):
        """If run with only one frame, set video_path=None and use run_one_frame() function"""
        self.box = Box()
        self.config = cfg
        self.inference_pipeline = self._init()

        # If the video_path and image_path are all included, it will use video_path
        video_path = None
        image_path = None

        for arg in LineDetectionDemo.required_args:
            if arg in kwargs:
                if arg == "video_path":
                    video_path = arg
                elif arg == "image_path":
                    image_path = arg

        if not image_path and not video_path:
            raise ValueError("At least one arguments video_path or image_path must be included.")

        self.box.update(**kwargs)

    def _init(self):
        model = MobileV2_MLSD_Tiny()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_pipeline = InferencePipeline(
            model=model, device=device, model_weight_path=self.config.model_weight_path
        )
        return inference_pipeline

    def run(
        self,
        image_save_name: str = "sample_image",
        save_dir: str = "./src/workdir/experiments/output",
        is_saving: bool = True,
    ) -> List[np.ndarray]:
        """Run demo"""
        if "video_path" in self.box:
            cap = cv2.VideoCapture(self.box.video_path)
            if cap.isOpened() is False:
                raise ValueError("Error opening video file! Please check the video.")
        else:
            # Run with one image
            cap = None

        outputs = []
        if not cap:
            _, output = self.inference_pipeline.run(
                image_path=self.box.image_path,
                image_save_name=image_save_name,
                save_dir=save_dir,
                is_saving=is_saving,
            )
            outputs.append(output)
            return outputs
        else:
            i = -1
            while cap.isOpened():
                ret, frame = cap.read()
                i += 1
                if ret is True:
                    _, output = self.inference_pipeline.run(
                        image=frame,
                        image_save_name=f"{image_save_name}_{i}",
                        save_dir=save_dir,
                        is_saving=is_saving,
                    )
                    outputs.append(output)
                else:
                    break
            return outputs

    def run_one_frame(
        self,
        frame: np.ndarray,
        image_save_name: str = "sample_image",
        save_dir: str = "./src/workdir/experiments/output",
        is_saving: bool = True,
    ) -> List[np.ndarray]:
        outputs = []
        _, output = self.inference_pipeline.run(
            image=frame,
            image_save_name=image_save_name,
            save_dir=save_dir,
            is_saving=is_saving,
        )
        outputs.append(output)
        return outputs
