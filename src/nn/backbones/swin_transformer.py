from torch import Tensor

from .base import BackboneBase


class SwinTransformerBackbone(BackboneBase):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.forward_features(x)
        if len(x.shape) == 2:
            b, c = x.shape
            x = x.reshape((b, c, 1, 1))
        return x

    @property
    def out_features(self) -> int:
        return self.model.num_features
