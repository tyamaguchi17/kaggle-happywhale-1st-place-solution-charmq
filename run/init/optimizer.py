from omegaconf import DictConfig
from torch.optim import SGD, Adam, AdamW
from torch_optimizer import AdaBelief, RAdam


def init_optimizer_from_config(cfg: DictConfig, params, return_cls=True):
    if cfg.type == "adam":
        opt_cls = Adam
        kwargs = {
            "lr": cfg.lr,
            "betas": (cfg.beta1, cfg.beta2),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
            "amsgrad": cfg.amsgrad,
        }
    elif cfg.type == "adamw":
        opt_cls = AdamW
        kwargs = {
            "lr": cfg.lr,
            "betas": (cfg.beta1, cfg.beta2),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
            "amsgrad": cfg.amsgrad,
        }
    elif cfg.type == "sgd":
        opt_cls = SGD
        kwargs = {
            "lr": cfg.lr,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
        }
    elif cfg.type == "radam":
        opt_cls = RAdam
        kwargs = {
            "lr": cfg.lr,
            "betas": (cfg.beta1, cfg.beta2),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        }
    elif cfg.type == "adabelief":
        opt_cls = AdaBelief
        kwargs = {
            "lr": cfg.lr,
            "betas": (cfg.beta1, cfg.beta2),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        }
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.type}")

    kwargs["params"] = params
    if return_cls:
        return opt_cls, kwargs
    else:
        return kwargs
