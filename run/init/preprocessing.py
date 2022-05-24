from typing import List, Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig


class Preprocessing:
    def __init__(
        self,
        aug_cfg: Optional[DictConfig],
        mean: List[float],
        std: List[float],
        h_resize_to: int,
        w_resize_to: int,
    ):
        if aug_cfg is None:
            self.aug_cfg = None
        else:
            self.aug_cfg = aug_cfg.copy()
        self.mean = mean.copy()
        self.std = std.copy()
        self.h_resize_to = h_resize_to
        self.w_resize_to = w_resize_to

    def get_train_transform(self) -> A.Compose:

        cfg = self.aug_cfg

        if cfg.use_aug:
            transforms = [
                A.Resize(self.h_resize_to, self.w_resize_to),
                A.Affine(
                    rotate=(-cfg.rotate, cfg.rotate),
                    translate_percent=(0.0, cfg.translate),
                    shear=(-cfg.shear, cfg.shear),
                    p=cfg.p_affine,
                ),
                A.RandomResizedCrop(
                    self.h_resize_to,
                    self.w_resize_to,
                    scale=(cfg.crop_scale, 1.0),
                    ratio=(cfg.crop_l, cfg.crop_r),
                ),
                A.ToGray(p=cfg.p_gray),
                A.GaussianBlur(blur_limit=(3, 7), p=cfg.p_blur),
                A.GaussNoise(p=cfg.p_noise),
                A.Downscale(scale_min=0.5, scale_max=0.5, p=cfg.p_downscale),
                A.RandomGridShuffle(grid=(2, 2), p=cfg.p_shuffle),
                A.Posterize(p=cfg.p_posterize),
                A.RandomBrightnessContrast(p=cfg.p_bright_contrast),
                A.Cutout(p=cfg.p_cutout),
                A.RandomSnow(p=cfg.p_snow),
                A.RandomRain(p=cfg.p_rain),
                A.RandomSunFlare(src_radius=40, p=cfg.p_sun),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(transpose_mask=True),
            ]
        else:
            transforms = [
                # Targets: image, mask, bboxes, keypoints
                A.Resize(self.h_resize_to, self.w_resize_to, p=1),
                # Targets: image
                A.Normalize(mean=self.mean, std=self.std),
                # Targets: image, mask
                ToTensorV2(transpose_mask=True),
            ]
        return A.Compose(transforms)

    def get_val_transform(self) -> A.Compose:
        transforms = [
            # Targets: image, mask, bboxes, keypoints
            A.Resize(self.h_resize_to, self.w_resize_to, p=1),
            # Targets: image
            A.Normalize(mean=self.mean, std=self.std),
            # Targets: image, mask
            ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(transforms)

    def get_test_transform(self) -> A.Compose:
        transforms = [
            # Targets: image, mask, bboxes, keypoints
            A.Resize(self.h_resize_to, self.w_resize_to, p=1),
            # Targets: image
            A.Normalize(mean=self.mean, std=self.std),
            # Targets: image, mask
            ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(transforms)
