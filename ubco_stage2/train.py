import lightning.pytorch as pl
import torch
import os

from ubco_stage2.callbacks import load_logger_and_callbacks

from ubco_stage2.mil_model.datamodule import CustomDataModule
from ubco_stage2.mil_model.module import CustomModule

def train(
        config,
):
    # # Set seed (not recommended during experimentation by @philippsinger)
    # pl.seed_everything(0, workers=True)

    # Set torch cache location
    torch.hub.set_dir(config.torch_cache)

    # Limit CPU if doing debug run
    if config.fast_dev_run == True:
        config.num_workers = 1

    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run = config.fast_dev_run,
        metrics = {
            "val_loss": "min", 
            "train_loss": "min",
            "val_acc": "last",
            "train_acc": "max",
            },
        overfit_batches = config.overfit_batches,
        no_wandb = config.no_wandb,
        project = config.project,
    )

    data_module = CustomDataModule(
        data_dir=config.data_dir,
        tile_size=config.tile_size,
        num_tiles=config.num_tiles,
        top_n_imgs=config.top_n_imgs,
        min_n_imgs=config.min_n_imgs,
        val_fold=config.val_fold,
        val_tta=config.val_tta,
        train_all_data=config.train_all_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    module = CustomModule(
        data_dir=config.data_dir,
        fast_dev_run=config.fast_dev_run,
        lr=config.lr,
        model_name=config.model_name,
        pretrained_weights=config.pretrained_weights,
        num_classes=config.num_classes,
        label_smoothing=config.label_smoothing,
        epochs=config.epochs,
        scheduler=config.scheduler,
        save_weights=config.save_weights,
        val_fold=config.val_fold,
        weight_decay=config.weight_decay,
        ham_pct=config.ham_pct,
        tile_size=config.tile_size,
    )

    # Trainer Args: https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        benchmark=True, # set to True if input size does not change (increases speed)
        fast_dev_run=config.fast_dev_run,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        overfit_batches=config.overfit_batches,
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        enable_checkpointing=False,
        gradient_clip_val=2.0,
    )

    if not config.valid_only:
        trainer.fit(module, datamodule=data_module)
        trainer.validate(module, datamodule=data_module)
    else:
        trainer.validate(module, datamodule=data_module)
    return