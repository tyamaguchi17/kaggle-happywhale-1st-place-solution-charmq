import math
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.optim.optimizer import Optimizer


class WarmupCosineLambda:
    def __init__(
        self,
        warmup_steps: int,
        cycle_steps: int,
        decay_scale: float,
        exponential_warmup: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(
                    self.decay_scale, -epoch / self.warmup_steps
                )
            ratio = epoch / self.warmup_steps
        else:
            ratio = (
                1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)
            ) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio


def init_scheduler_from_config(cfg: DictConfig, optimizer: Optimizer) -> Optional:
    if cfg.type is None:
        return None
    elif cfg.type == "step_lr":
        return StepLR(optimizer, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay_rate)
    elif cfg.type == "exponential_lr":
        return ExponentialLR(optimizer, gamma=cfg.lr_decay_rate)
    elif cfg.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_decay_rate,
            patience=cfg.patience,
            verbose=True,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == "cosine_annealing_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min
        )
    elif cfg.type == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    elif cfg.type == "cosine_warmup":
        warmup_steps = cfg.max_epochs * cfg.warmup_steps_ratio
        cycle_steps = cfg.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, cfg.lr_decay_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")
