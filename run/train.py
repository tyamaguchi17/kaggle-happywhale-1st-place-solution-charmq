import logging
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from run.pl_model import PLModel

logger = logging.getLogger(__name__)


def main(cfg: DictConfig, pl_model: type) -> Path:
    seed_everything(cfg.training.seed)
    out_dir = Path(cfg.out_dir).resolve()

    if cfg.test_model is not None:
        # Only run test with the given model weights
        is_test_mode = True
    else:
        # Run full training
        is_test_mode = False

    # init experiment logger
    if not cfg.training.use_wandb or is_test_mode:
        pl_logger = False
    else:
        pl_logger = WandbLogger(
            project=cfg.training.project_name,
            save_dir=str(out_dir),
            name=Path(out_dir).name,
        )

    # init lightning model
    model = pl_model(cfg)

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.training.monitor,
        mode=cfg.training.monitor_mode,
        save_top_k=1,
        save_last=True,
    )

    # init trainer
    def _init_trainer(resume=True):
        resume_from = cfg.training.resume_from if resume else None
        return Trainer(
            # env
            default_root_dir=str(out_dir),
            gpus=cfg.training.num_gpus,
            accelerator="ddp",
            precision=16 if cfg.training.use_amp else 32,
            # training
            fast_dev_run=cfg.training.debug,  # run only 1 train batch and 1 val batch
            weights_summary="top" if cfg.training.debug else None,
            max_epochs=cfg.training.epoch,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            callbacks=[checkpoint_cb],
            logger=pl_logger,
            resume_from_checkpoint=resume_from,
            num_sanity_val_steps=0 if is_test_mode else 2,
            sync_batchnorm=True,
        )

    trainer = _init_trainer()

    if cfg.training.resume_from is not None:
        ckpt = torch.load(cfg.training.resume_from, map_location="cpu")
        initial_best_score = ckpt["callbacks"][ModelCheckpoint]["best_model_score"]
        initial_best_score = initial_best_score.detach().cpu().numpy()
        initial_best_model = ckpt["callbacks"][ModelCheckpoint]["best_model_path"]
        del ckpt
        logger.info(
            f"Initial best model ({initial_best_score:.4f}): {initial_best_model}"
        )

    if is_test_mode:
        trainer.test(model)
    else:
        trainer.fit(model)

        if cfg.training.resume_from is None:
            trainer.test()  # test with the best checkpoint
        else:
            current_best_score = (
                trainer.checkpoint_callback.best_model_score.detach().cpu().numpy()
            )
            current_larger_than_initial = current_best_score > initial_best_score
            mode = trainer.checkpoint_callback.mode
            best_updated = (mode == "max" and current_larger_than_initial) or (
                mode == "min" and not current_larger_than_initial
            )
            if best_updated:
                best_ckpt = trainer.checkpoint_callback.best_model_path
                logger.info("The best model is updated.")
            else:
                best_ckpt = initial_best_model
                logger.info("The best model isn't changed.")

            current_epoch = trainer.current_epoch
            try:
                state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
            except FileNotFoundError:
                time.sleep(30)
                state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            trainer = _init_trainer(resume=False)
            trainer.current_epoch = current_epoch
            logger.info(f"Testing with the best ckpt: {best_ckpt}")
            trainer.test(model)

        # extract weights and save
        if trainer.global_rank == 0:
            weights_path = str(Path(checkpoint_cb.dirpath) / "model_weights.pth")
            logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(model.forwarder.model.state_dict(), weights_path)

    # return path to checkpoints directory
    if checkpoint_cb.dirpath is not None:
        return Path(checkpoint_cb.dirpath)


def prepare_env() -> None:
    # Disable PIL's debug logs
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # move to original directory
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # set PYTHONPATH if not set for possible launching of DDP processes
    os.environ.setdefault("PYTHONPATH", ".")


@hydra.main(config_path="conf", config_name="config")
def entry(cfg: DictConfig) -> None:
    prepare_env()
    main(cfg, PLModel)


if __name__ == "__main__":
    entry()
