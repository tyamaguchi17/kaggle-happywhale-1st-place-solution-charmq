from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
        phase: str,
        tta: bool = False,
    ):
        self.base = base
        self.transform = transform
        self.label_to_samples = base.label_to_samples
        self.label_names = base.label_names

    def __len__(self) -> int:
        return len(self.base)

    def apply_transform(self, data):

        img = data.pop("image")

        if img.ndim == 2:  # convert grayscale img to rgb
            img = np.asarray([img, img, img]).transpose((1, 2, 0))  # (H, W, 3)

        if "mask" in data:
            mask = data.pop("mask")
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]  # (3, H, W)
            mask = transformed["mask"] / 255.0
            data["image"] = torch.cat([img, mask.unsqueeze(0)])
        else:
            transformed = self.transform(image=img)
            data["image"] = transformed["image"]  # (3, H, W)

        if "image2" in data:
            img = data.pop("image2")

            if img.ndim == 2:  # convert grayscale img to rgb
                img = np.asarray([img, img, img]).transpose((1, 2, 0))  # (H, W, 3)

            if "mask2" in data:
                mask = data.pop("mask2")
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]  # (3, H, W)
                mask = transformed["mask"] / 255.0
                img = torch.cat([img, mask.unsqueeze(0)])
            else:
                transformed = self.transform(image=img)
                img = transformed["image"]  # (3, H, W)

            data["image"] = torch.cat([data["image"], img])

        return data

    def __getitem__(self, index: int):
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data
