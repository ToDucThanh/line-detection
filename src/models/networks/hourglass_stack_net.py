import time

import torch
import torch.nn as nn

from src.models.networks.modules import (
    Bottleneck1D_h,
    Bottleneck1D_v,
    Bottleneck2D,
    BottleneckLine,
    Hourglass,
    ModuleBase,
    MultitaskHead,
)
from src.utils.placeholder import M

block_dicts = {
    "Bottleneck2D": Bottleneck2D,
    "Bottleneck1D_v": Bottleneck1D_v,
    "Bottleneck1D_h": Bottleneck1D_h,
    "BottleneckLine": BottleneckLine,
}


class HourglassStackNet(ModuleBase):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(
        self,
        head: ModuleBase,
        depth: int,
        num_stacks: int,
        num_blocks: int,
        num_classes: int,
    ):
        super(HourglassStackNet, self).__init__()

        branch_blocks = []
        for key in M.branch_blocks:
            branch_blocks.append(block_dicts[key])

        self.inplanes = M.inplanes
        self.num_feats = self.inplanes * Bottleneck2D.expansion
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(Bottleneck2D, self.inplanes, 1)
        self.layer2 = self._make_residual(Bottleneck2D, self.inplanes, 1)
        self.layer3 = self._make_residual(Bottleneck2D, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * Bottleneck2D.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        merge_fc = []
        for i in range(num_stacks):
            sub_hg, sub_res, sub_fc = [], [], []
            for bb in branch_blocks:
                sub_hg.append(Hourglass(bb, num_blocks, self.num_feats, depth))
                sub_res.append(self._make_residual(bb, self.num_feats, num_blocks))
                sub_fc.append(self._make_fc(ch, ch))
            hg.append(nn.ModuleList(sub_hg))
            res.append(nn.ModuleList(sub_res))
            fc.append(nn.ModuleList(sub_fc))
            merge_fc.append(self._make_fc(int(ch * len(sub_fc)), ch))

            score.append(head(ch, num_classes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.merge_fc = nn.ModuleList(merge_fc)

        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        bn = nn.BatchNorm2d(outplanes)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        extra_info = {
            "time_front": 0.0,
            "time_stack0": 0.0,
            "time_stack1": 0.0,
        }

        t = time.time()
        out = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        extra_info["time_front"] = time.time() - t

        for i in range(self.num_stacks):
            t = time.time()
            feat = []
            for j in range(len(self.hg[i])):
                y = self.hg[i][j](x)
                y = self.res[i][j](y)
                y = self.fc[i][j](y)
                feat.append(y)

            y = self.merge_fc[i](torch.cat(feat, dim=1))
            score = self.score[i](y)
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

            extra_info[f"time_stack{i}"] = time.time() - t

        return out[::-1], y, extra_info

    @classmethod
    def get_model(cls, kwargs):
        model = cls(
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            depth=kwargs["depth"],
            num_stacks=kwargs["num_stacks"],
            num_blocks=kwargs["num_blocks"],
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
        return model
