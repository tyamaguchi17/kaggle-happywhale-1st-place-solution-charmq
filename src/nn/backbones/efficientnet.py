from timm.models.efficientnet import EfficientNet
from torch import Tensor

from .base import BackboneBase


class EfficientNetBackbone(BackboneBase):
    def __init__(self, model: EfficientNet) -> None:
        assert isinstance(model, EfficientNet)
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.forward_features(x)
        return x

    @property
    def out_features(self) -> int:
        return self.model.classifier.in_features
