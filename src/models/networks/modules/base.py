from abc import (
    ABC,
    abstractmethod,
)

import torch.nn as nn


class ModuleBase(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        """Initialize all layers needed"""
        super(ModuleBase, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_model(cls, *args, **kwargs):
        return cls(*args, **kwargs)
