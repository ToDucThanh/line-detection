from src.models.pipeline.mlsd.train import TrainingPipeline
from src.utils.placeholder import Box

if __name__ == "__main__":
    cfg = Box()
    cfg.update(
        cfg.from_yaml(
            filename="src/models/model_cfg/mobilev2_mlsd_tiny_512_base2_bsize24.yaml"
        )
    )
    training_pipeline = TrainingPipeline(cfg=cfg)
    score = training_pipeline.run()
    print(score)
