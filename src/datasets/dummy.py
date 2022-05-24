from typing import Dict, Optional

import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Dummy dataset, mainly for testing.

    Returned image size, image values, labels and (if `with_mask` is set) mask are all
    randomly generated.

    Args:
        length: Dataset length.
        seed: Random seed. If None, determine the seed randomly.
        with_mask: If True, also return "mask" of the same size as the image.
    """

    CLASS_NAMES = ["DummyA", "DummyB", "DummyC"]

    def __init__(
        self, length: int, seed: Optional[int] = None, with_mask: bool = False
    ) -> None:
        super().__init__()
        self.length = length
        self.rng = np.random.RandomState(seed)
        self.with_mask = with_mask

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        height = self.rng.randint(128, 256)
        width = self.rng.randint(128, 256)
        x = self.rng.randint(256, size=(height, width)).astype(np.uint8)
        y = self.rng.randint(2, size=(3,)).astype(np.float32)  # 3 classes, [0, 1]
        item = {"img": x, "label": y}
        if self.with_mask:
            # -1, 0, or 1
            item["mask"] = self.rng.randint(3, size=x.shape).astype(np.float32) - 1
        return item
