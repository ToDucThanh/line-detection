from src.utils.placeholder import M

from .base import (
    ModuleBase,
    nn,
)


class Bottleneck1D_h(ModuleBase):
    expansion = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super(Bottleneck1D_h, self).__init__()

        self.ks = (1, M.line_kernel)
        self.padding = (0, int(M.line_kernel / 2))

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=self.ks, stride=stride, padding=self.padding
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck1D_h.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
