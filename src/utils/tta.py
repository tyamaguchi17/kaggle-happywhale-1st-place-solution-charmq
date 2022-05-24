import itertools
from typing import Callable, List, Sequence, Union

import torch
from torch import Tensor


class Transform:
    def __init__(self, flip_h: bool, n_rot90: int) -> None:
        self.flip_h = flip_h
        self.n_rot90 = n_rot90

    def __call__(self, x: Tensor) -> Tensor:
        assert (
            len(x.shape) == 4
        ), "`x` must have 4 dimensions: (batch, channel, height, width)"
        if self.flip_h:
            x = torch.flip(x, [3])
        x = x.rot90(self.n_rot90, dims=[2, 3])
        return x

    def inverse(self, x: Tensor) -> Tensor:
        """Returns inverse transformation of `__call__`."""
        assert (
            len(x.shape) == 4
        ), "`x` must have 4 dimensions: (batch, channel, height, width)"
        x = x.rot90(-self.n_rot90, dims=[2, 3])
        if self.flip_h:
            x = torch.flip(x, [3])
        return x


class TestTimeAugmentor:
    __test__ = False  # exclude from unit tests

    def __init__(self, flip_h: bool, rot90: bool, rot180: bool) -> None:
        ns_rot90 = [0]  # no rotation
        if rot90:
            ns_rot90 += [1, 3]  # 90- and 270-degree rotations
        if rot180:
            ns_rot90 += [2]  # 180-degree rotation

        augmentations = {
            "flip_h": [False, True] if flip_h else [False],
            "n_rot90": ns_rot90,
        }

        keys = augmentations.keys()
        values = list(augmentations.values())

        self.transforms = []
        for c in itertools.product(*values):
            self.transforms.append(Transform(**dict(zip(keys, c))))

    def get_inputs(self, x: Tensor) -> List[Tensor]:
        """Apply transforms to the tensor and return them as a list."""
        return [t(x) for t in self.transforms]

    @staticmethod
    def run(
        model: Callable[[Tensor], Union[Tensor, Sequence[Tensor]]],
        xs: Sequence[Tensor],
    ) -> List[Tensor]:
        ys = []
        for x in xs:
            y = model(x)  # assuming `y` is a sequence of Tensors
            if isinstance(y, Tensor):
                y = (y,)
            ys.append(y)

        ret = []
        for i in range(len(ys[0])):
            tensors = [ys[j][i] for j in range(len(ys))]
            ret.append(torch.stack(tensors))
        return ret
