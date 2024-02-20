# from src.models.pipeline.mlsd.train import TrainingPipeline
import torch

from src.models.networks.mobilev2_mlsd_tiny_net import MobileV2_MLSD_Tiny
from src.models.pipeline.mlsd.inference import InferencePipeline
from src.utils.placeholder import Box

if __name__ == "__main__":
    cfg = Box()
    cfg.update(
        cfg.from_yaml(
            filename="src/models/model_cfg/mobilev2_mlsd_tiny_512_base2_bsize24.yaml"
        )
    )
    # training_pipeline = TrainingPipeline(cfg=cfg)
    # score = training_pipeline.run()
    # print(score)
    model = MobileV2_MLSD_Tiny()
    model_path = cfg.model.weight_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_pipeline = InferencePipeline(
        model=model, device=device, model_weight_path=model_path
    )
    inference_pipeline.run(
        image_path="src/data/v1.1/train/00030077.jpg",
        image_save_name="00030077_output",
        save_dir="/Users/mac/Main/FPT/DN-airport/research/line-detection/combination/src/workdir/experiments/output",
        is_saving=True,
    )
