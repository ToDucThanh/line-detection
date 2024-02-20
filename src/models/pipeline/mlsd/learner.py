import os
from typing import Union

import numpy as np
import torch
import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.models.loss import LineSegmentLoss
from src.models.metric import (
    AP,
    F1_score_128,
    msTPFP,
)
from src.utils.decode import deccode_lines_TP
from src.utils.meter import AverageMeter
from src.utils.txt_logger import TxtLogger


class Simple_MLSD_Learner:
    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler,
        logger: TxtLogger,
        save_dir: str,
        log_steps=100,
        device: Union[torch.device, str] = "cpu",
        gradient_accum_steps=1,
        max_grad_norm=100.0,
        batch_to_model_inputs_fn=None,
        early_stop_n=4,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device = device
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0

        self.input_size = self.cfg.datasets.input_size
        self.loss_fn = LineSegmentLoss(cfg)
        self.epo = 0

    def step(self, step_n, batch_data: dict):
        imgs = batch_data["xs"].to(self.device)
        label = batch_data["ys"].to(self.device)
        outputs = self.model(imgs)
        loss_dict = self.loss_fn(
            outputs,
            label,
            batch_data["gt_lines_tensor_512_list"],
            batch_data["sol_lines_512_all_tensor_list"],
        )
        loss = loss_dict["loss"]
        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        return loss, loss_dict

    def val(self, model, val_dataloader: DataLoader):
        thresh = self.cfg.decode.score_thresh
        topk = self.cfg.decode.top_k
        min_len = self.cfg.decode.len_thresh

        model = model.eval()
        sap_thresh = 10
        data_iter = tqdm.tqdm(val_dataloader)
        f_scores = []
        recalls = []
        precisions = []

        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for batch_data in data_iter:
            imgs = batch_data["xs"].to(self.device)
            label = batch_data["ys"].to(self.device)
            batch_outputs = model(imgs)

            label = label[:, 7:, :, :]
            batch_outputs = batch_outputs[:, 7:, :, :]

            for outputs, gt_lines_512 in zip(batch_outputs, batch_data["gt_lines_512"]):
                gt_lines_512 = np.array(gt_lines_512, np.float32)

                outputs = outputs.unsqueeze(0)

                center_ptss, pred_lines, _, scores = deccode_lines_TP(
                    outputs, thresh, min_len, topk, 3
                )

                pred_lines = pred_lines.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()

                pred_lines_128 = 128 * pred_lines / (self.input_size / 2)

                gt_lines_128 = gt_lines_512 / 4
                fscore, recall, precision = F1_score_128(
                    pred_lines_128.tolist(), gt_lines_128.tolist(), thickness=3
                )
                f_scores.append(fscore)
                recalls.append(recall)
                precisions.append(precision)

                tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)

                n_gt += gt_lines_128.shape[0]
                tp_list.append(tp)
                fp_list.append(fp)
                scores_list.append(scores)

        f_score = np.array(f_scores, np.float32).mean()
        recall = np.array(recalls, np.float32).mean()
        precision = np.array(precisions, np.float32).mean()

        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx]) / n_gt
        fp = np.cumsum(fp_list[idx]) / n_gt

        sAP = AP(tp, fp) * 100
        self.logger.write(
            "==>step: {}, f_score: {}, recall: {}, precision:{}, sAP10: {}\n ".format(
                self.global_step, f_score, recall, precision, sAP
            )
        )
        return {
            "fscore": f_score,
            "recall": recall,
            "precision": precision,
            "sAP10": sAP,
        }

    def train(
        self, train_dataloader: DataLoader, val_dataloader: DataLoader, epoches=100
    ):
        best_score = 0
        early_n = 0
        for self.epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_avg_center_loss = AverageMeter()
            train_avg_replacement_loss = AverageMeter()
            train_avg_line_seg_loss = AverageMeter()
            train_avg_junc_seg_loss = AverageMeter()

            train_avg_match_loss = AverageMeter()
            train_avg_match_rario = AverageMeter()

            data_iter = tqdm.tqdm(train_dataloader)
            for idx, batch in enumerate(data_iter):
                self.model.train()
                train_loss, loss_dict = self.step(step_n, batch)
                train_avg_loss.update(train_loss.item(), 1)

                train_avg_center_loss.update(loss_dict["center_loss"].item(), 1)
                train_avg_replacement_loss.update(
                    loss_dict["displacement_loss"].item(), 1
                )
                train_avg_line_seg_loss.update(loss_dict["line_seg_loss"].item(), 1)
                train_avg_junc_seg_loss.update(loss_dict["junc_seg_loss"].item(), 1)
                train_avg_match_loss.update(float(loss_dict["match_loss"]), 1)
                train_avg_match_rario.update(loss_dict["match_ratio"], 1)

                status = (
                    "[{0}] lr= {1:.6f} loss= {2:.3f} avg = {3:.3f},c: {4:.3f}, d: {5:.3f}, l: {6:.3f}, "
                    "junc:{7:.3f},m:{8:.3f},m_r:{9:.2f} ".format(
                        self.epo + 1,
                        self.scheduler.get_lr()[0],
                        train_loss.item(),
                        train_avg_loss.avg,
                        train_avg_center_loss.avg,
                        train_avg_replacement_loss.avg,
                        train_avg_line_seg_loss.avg,
                        train_avg_junc_seg_loss.avg,
                        train_avg_match_loss.avg,
                        train_avg_match_rario.avg,
                    )
                )
                data_iter.set_description(status)
                step_n += 1

            if self.epo > self.cfg.val.val_after_epoch:
                m = self.val(self.model, val_dataloader)
                fscore = m["sAP10"]
                if best_score < fscore:
                    early_n = 0
                    best_score = fscore
                    model_path = os.path.join(self.save_dir, "best.pth")
                    torch.save(self.model.state_dict(), model_path)
                else:
                    early_n += 1
                self.logger.write(
                    "epo: {}, steps: {} ,sAP10 : {:.4f} , best sAP10: {:.4f}".format(
                        self.epo, self.global_step, fscore, best_score
                    )
                )
                self.logger.write(str(m))
                self.logger.write("==" * 50)

                if early_n > self.early_stop_n:
                    print("early stopped!")
                    return best_score
            model_path = os.path.join(self.save_dir, "latest.pth")
            torch.save(self.model.state_dict(), model_path)
        return best_score
