from timm.models.resnet import ResNet
from torch import Tensor

from .base import BackboneBase


class ResNetBackbone(BackboneBase):
    def __init__(self, model: ResNet) -> None:
        assert isinstance(model, ResNet)
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.forward_features(x)
        return x

    @property
    def out_features(self) -> int:
        return self.model.fc.in_features
