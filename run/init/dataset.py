from typing import Dict

import pandas as pd
from omegaconf import DictConfig

from src.datasets.dummy import DummyDataset
from src.datasets.happy_whale import HappyWhaleDataset


def init_datasets_from_config(cfg: DictConfig):
    if cfg.type == "dummy":
        ds = DummyDataset(length=cfg.dummy_length, with_mask=cfg.with_mask)
        datasets = {"train": ds, "val": ds, "test": ds}
        # class_names = DummyDataset.CLASS_NAMES
    elif cfg.type == "happy_whale":
        datasets = get_happy_whale_dataset(
            num_folds=cfg.num_folds,
            test_fold=cfg.test_fold,
            val_fold=cfg.val_fold,
            seed=cfg.seed,
            num_records=cfg.num_records,
            phase=cfg.phase,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.type}")

    return datasets


def get_happy_whale_dataset(
    num_folds: int,
    test_fold: int,
    val_fold: int,
    seed: int = 2022,
    num_records: int = 0,
    phase: str = "train",
    cfg=None,
) -> Dict[str, HappyWhaleDataset]:

    df = HappyWhaleDataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        phase,
    )

    if cfg.pseudo_label_filename is not None:
        df_pl = HappyWhaleDataset.create_dataframe(
            num_folds,
            seed,
            num_records,
            phase,
            pseudo_label_filename=cfg.pseudo_label_filename,
        )
        df_pl = df_pl[df_pl["conf"] > cfg.pseudo_label_conf]

    if phase == "train":
        train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
        val_df = df[df["fold"] == val_fold]
        test_df = df[df["fold"] == test_fold]
        if cfg.pseudo_label_filename is not None:
            train_df = pd.concat([train_df, df_pl])

        train_dataset = HappyWhaleDataset(train_df, phase, cfg, crop_aug=cfg.crop_aug)
        val_dataset = HappyWhaleDataset(val_df, phase, cfg)
        test_dataset = HappyWhaleDataset(test_df, phase, cfg)
    elif phase == "valid":
        train_dataset = HappyWhaleDataset(df, "train", cfg)
        val_dataset = HappyWhaleDataset(df, "train", cfg)
        test_dataset = HappyWhaleDataset(df, "train", cfg)
    elif phase == "test":
        train_dataset = HappyWhaleDataset(df, phase, cfg)
        val_dataset = HappyWhaleDataset(df, phase, cfg)
        test_dataset = HappyWhaleDataset(df, phase, cfg)
    elif phase == "all":
        if cfg.pseudo_label_filename is not None:
            train_dataset = HappyWhaleDataset(pd.concat([df, df_pl]), "train", cfg)
        else:
            train_dataset = HappyWhaleDataset(df, "train", cfg)
        val_dataset = HappyWhaleDataset(df[:100], "train", cfg)
        test_dataset = HappyWhaleDataset(df[:100], "train", cfg)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    return datasets
