import torch

from src.utils.placeholder import M

from .base import (
    ModuleBase,
    nn,
)


class BottleneckLine(ModuleBase):
    expansion = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super(BottleneckLine, self).__init__()

        # self.ks = (M.line_kernel, 1)
        # self.padding = (int(M.line_kernel / 2), 0)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.conv2D = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.conv2v = nn.Conv2d(planes, planes, kernel_size=(M.line_kernel, 1), padding=(int(M.line_kernel / 2), 0))
        # self.conv2h = nn.Conv2d(planes, planes, kernel_size=(1, M.line_kernel), padding=(0, int(M.line_kernel / 2)))

        self.conv2, self.merge = self.build_line_layers(planes)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * BottleneckLine.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def build_line_layers(self, planes):
        layer = []
        if "s" in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))

        if "v" in M.line.mode:
            layer.append(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=(M.line_kernel, 1),
                    padding=(int(M.line_kernel / 2), 0),
                )
            )

        if "h" in M.line.mode:
            layer.append(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=(1, M.line_kernel),
                    padding=(0, int(M.line_kernel / 2)),
                )
            )

        assert len(layer) > 0

        if M.merge == "cat":
            merge = nn.Conv2d(planes * len(layer), planes, kernel_size=1)
        elif M.merge == "maxpool":
            ll = len(M.line.mode)
            merge = nn.MaxPool3d((ll, 1, 1), stride=(ll, 1, 1))
        else:
            raise ValueError()

        return nn.ModuleList(layer), merge

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        if M.merge == "cat":
            tt = torch.cat([conv(out) for conv in self.conv2], dim=1)
        elif M.merge == "maxpool":
            tt = torch.cat(
                [torch.unsqueeze(conv(out), 2) for conv in self.conv2], dim=2
            )
        else:
            raise ValueError()
        out = self.merge(tt)
        out = torch.squeeze(out, 2)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
