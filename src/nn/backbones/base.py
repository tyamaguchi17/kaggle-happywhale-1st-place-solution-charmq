import torch.nn as nn
from torch import Tensor


class BackboneBase(nn.Module):
    """Base class for a backbone.

    A backbone should take a tensor of shape (B, 3, H, W) as input and output features
    of shape (B, out_features, h, w).
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def out_features(self) -> int:
        raise NotImplementedError
