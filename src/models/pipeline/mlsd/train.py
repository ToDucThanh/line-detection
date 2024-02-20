import math
import os

import torch

from src.data.data_loader import LineDataloader
from src.models.networks.mobilev2_mlsd_net import (
    MobileV2_MLSD,
    MobileV2_MLSD_Large,
)
from src.models.optim import WarmupMultiStepLR
from src.utils.comm import (
    create_dir,
    setup_seed,
)
from src.utils.placeholder import Box
from src.utils.txt_logger import TxtLogger

from .learner import Simple_MLSD_Learner


class TrainingPipeline:
    def __init__(self, cfg: Box):
        self.cfg = cfg

    def build_model(self):
        model_name = self.cfg.model.model_name
        if model_name == "mobilev2_mlsd":
            m = MobileV2_MLSD(self.cfg)
            return m
        if model_name == "mobilev2_mlsd_large":
            m = MobileV2_MLSD_Large(self.cfg)
            return m
        raise NotImplementedError("{} no such model!".format(model_name))

    def _lr_lambda_fn(
        self, step: int, warmup_steps, start_step, end_step, min_lr_scale, n_t
    ):
        if step < warmup_steps:
            return 0.9 * step / warmup_steps + 0.1
        elif step < start_step:
            return 1.0
        else:
            cos_val = n_t * (
                1 + math.cos(math.pi * (step - start_step) / (end_step - start_step))
            )
            return min_lr_scale if cos_val < min_lr_scale else cos_val

    def run(self):
        train_loader = LineDataloader.get_dataloader(cfg=self.cfg)
        valid_loader = LineDataloader.get_dataloader(cfg=self.cfg, is_train=False)
        model = self.build_model()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if os.path.exists(self.cfg.train.load_from):
            print("load from: ", self.cfg.train.load_from)
            model.load_state_dict(
                torch.load(self.cfg.train.load_from, map_location=device), strict=False
            )

        if self.cfg.train.milestones_in_epo:
            ns = len(train_loader)
            milestones = []
            for m in self.cfg.train.milestones:
                milestones.append(m * ns)
            self.cfg.train.milestones = milestones

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        if self.cfg.train.use_step_lr_policy:
            lr_scheduler = WarmupMultiStepLR(
                optimizer,
                milestones=self.cfg.train.milestones,
                gamma=self.cfg.train.lr_decay_gamma,
                warmup_iters=self.cfg.train.warmup_steps,
            )
        else:
            warmup_steps = 5 * len(train_loader)  ## 5 epoch warmup
            min_lr_scale = 0.0001
            start_step = 70 * len(train_loader)
            end_step = 150 * len(train_loader)
            n_t = 0.5

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: self.lr_lambda_fn(
                    step, warmup_steps, start_step, end_step, min_lr_scale, n_t
                ),
            )
        create_dir(self.cfg.train.save_dir)
        logger = TxtLogger(self.cfg.train.save_dir + "/train_logger.txt")
        learner = Simple_MLSD_Learner(
            self.cfg,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            logger=logger,
            save_dir=self.cfg.train.save_dir,
            log_steps=self.cfg.train.log_steps,
            device=device,
            gradient_accum_steps=1,
            max_grad_norm=1000.0,
            batch_to_model_inputs_fn=None,
            early_stop_n=self.cfg.train.early_stop_n,
        )
        learner.train(
            train_loader, valid_loader, epoches=self.cfg.train.num_train_epochs
        )
