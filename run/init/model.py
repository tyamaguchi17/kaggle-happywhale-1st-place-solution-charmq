from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.datasets.happy_whale import HappyWhaleDataset
from src.nn.backbone import load_backbone
from src.nn.backbones.base import BackboneBase
from src.nn.heads.arc_face import ArcAdaptiveMarginProduct, ChannelWiseGeM, GeM
from src.utils.checkpoint import get_weights_to_load

logger = getLogger(__name__)


def init_model_from_config(cfg: DictConfig, pretrained: bool):
    model = nn.Sequential()

    if cfg.head.type == "adaptive_arcface":

        def get_forward_features():
            backbone = init_backbone(cfg, pretrained=pretrained)
            forward_features = nn.Sequential()
            forward_features.add_module("backbone", backbone)
            if cfg.pool.type == "adaptive":
                forward_features.add_module("pool", nn.AdaptiveAvgPool2d((1, 1)))
                forward_features.add_module("flatten", nn.Flatten())
            elif cfg.pool.type == "gem":
                forward_features.add_module(
                    "pool", GeM(p=cfg.pool.p, p_trainable=cfg.pool.p_trainable)
                )
                forward_features.add_module("flatten", nn.Flatten())
            elif cfg.pool.type == "gem_ch":
                forward_features.add_module(
                    "pool",
                    ChannelWiseGeM(
                        dim=backbone.out_features,
                        p=cfg.pool.p,
                        requires_grad=cfg.pool.p_trainable,
                    ),
                )
                forward_features.add_module("flatten", nn.Flatten())

            embedding_size = backbone.out_features
            if cfg.embedding_size > 0:
                embedding_size = cfg.embedding_size
                forward_features.add_module(
                    "linear",
                    nn.Linear(backbone.out_features, embedding_size, bias=True),
                )
            if cfg.use_bn:
                forward_features.add_module("normalize", nn.BatchNorm1d(embedding_size))
                forward_features.add_module("relu", torch.nn.PReLU())
            return forward_features, embedding_size

        forward_features, embedding_size = get_forward_features()
        model.add_module("forward_features", forward_features)
        if cfg.backbone2:
            forward_features2, embedding_size2 = get_forward_features()
            model.add_module("forward_features2", forward_features2)
        else:
            embedding_size2 = 0

        if cfg.species_embedding_size > 0:
            species_embedding_size = cfg.species_embedding_size
            model.add_module(
                "species_embedding",
                nn.Embedding(cfg.output_dim_species, species_embedding_size),
            )
        else:
            species_embedding_size = 0

        # dynamic margin
        root = HappyWhaleDataset.ROOT_PATH
        df = pd.read_csv(root / "train.csv")
        id_class_nums = df.individual_id.value_counts().sort_index().values
        df.species.replace(
            {
                "globis": "short_finned_pilot_whale",
                "pilot_whale": "short_finned_pilot_whale",
                "kiler_whale": "killer_whale",
                "bottlenose_dolpin": "bottlenose_dolphin",
            },
            inplace=True,
        )
        species_class_nums = df.species.value_counts().sort_index().values
        margins = (
            np.power(id_class_nums, cfg.head.margin_power_id) * cfg.head.margin_coef_id
            + cfg.head.margin_cons_id
        )
        margins_species = (
            np.power(species_class_nums, cfg.head.margin_power_species)
            * cfg.head.margin_coef_species
            + cfg.head.margin_cons_species
        )
        if margins.shape[0] * 2 == cfg.output_dim:
            margins = np.hstack([margins, margins])

        head = ArcAdaptiveMarginProduct(
            embedding_size + embedding_size2 + species_embedding_size,
            cfg.output_dim,
            margins=margins,
            s=cfg.head.s,
            k=cfg.head.k,
            initialization=cfg.head.init,
        )
        model.add_module("head", head)
        head_species = ArcAdaptiveMarginProduct(
            embedding_size + embedding_size2 + species_embedding_size,
            cfg.output_dim_species,
            margins=margins_species,
            s=cfg.head.s_species,
            k=cfg.head.k_species,
            initialization=cfg.head.init_species,
        )
        model.add_module("head_species", head_species)

    else:
        raise ValueError(f"Unknown head type: {cfg.head.type}")

    if cfg.restore_path is not None:
        logger.info(f'Loading weights from "{cfg.restore_path}"...')
        ckpt = torch.load(cfg.restore_path, map_location="cpu")
        model_dict = get_weights_to_load(model, ckpt)
        model.load_state_dict(model_dict, strict=True)

    return model


def init_backbone(cfg: DictConfig, pretrained: bool) -> BackboneBase:

    backbone = load_backbone(
        base_model=cfg.base_model,
        pretrained=pretrained,
        in_chans=cfg.in_chans,
    )
    if cfg.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone
