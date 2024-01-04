import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchinfo

import os
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from ubco_stage2.mil_model.model import MILModel

class CustomModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: str,
        fast_dev_run: bool,
        lr: float,
        model_name: str,
        pretrained_weights: str,
        num_classes: int,
        epochs: int,
        scheduler: str,
        label_smoothing: float,
        save_weights: bool,
        val_fold: int,
        weight_decay: float,
        ham_pct: float,
        tile_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()
        self.label2id = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4}
        self.id2label = {0: 'CC', 1: 'EC', 2: 'HGSC', 3: 'LGSC', 4: 'MC'}
        self.val_outputs = []
        self.model_id = np.random.randint(0, 10_000)

    def _init_model(self):
        model = MILModel(
            backbone = self.hparams.model_name,
        )
                        
        if self.hparams.pretrained_weights != "":
            print("Loading trained backbone.. ")
            weights_path = os.path.join(self.hparams.data_dir, "models/", self.hparams.pretrained_weights)
            model.load_state_dict(torch.load(weights_path))

        # torchinfo.summary(model)
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(self.trainer.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "Constant":
            # Hacky fix to keep constant LR
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.max_epochs,
                eta_min = 1e-7,
                )
        elif self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.trainer.max_epochs,
                eta_min = 1e-7,
                )
        else:
            raise ValueError(f"{self.hparams.scheduler} is not a valid scheduler.")
        
    def lr_scheduler_step(self, scheduler, optimizer_idx) -> None:
        scheduler.step()
        return
    
    def _init_loss_fn(self):
        return torch.nn.CrossEntropyLoss(
            label_smoothing=self.hparams.label_smoothing,
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def forward(self, x, ham_pct=0.0):
        return self.model(x, ham_pct)
    
    def _shared_step(self, batch, stage, batch_idx):
        x, y, is_tma, img_id = batch

        if stage == "val":
            y_logits = self(x, ham_pct=0)
            self.val_outputs.append({
                "logit": F.softmax(y_logits, dim=1), 
                "label": y, 
                "is_tma": is_tma,
                })
            
            # # Writes predictions to OOF (Should move this to on_validation_end() function)
            # torch.save(y_logits[0, ...], os.path.join(self.hparams.data_dir, "preds", img_id[0] + ".pt"))
        
        else:
            y_logits = self(x, ham_pct=self.hparams.ham_pct)
        loss = self.loss_fn(y_logits, y)
        self._log(stage, loss, batch_size=len(x))
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        self._shared_step(batch, "val", batch_idx)
        return
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)
    
    def _log(self, stage, loss, batch_size) -> None:
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return

    def _accuracy_for_each_class(self, labels, preds, suffix="") -> dict:
        """
        Helper function to compute by-class 
        accuracy.

        Args: 
            labels (np.array): Labels
            preds (np.array): Predictions

        Returns:
            metrics (dict): Accuracy by class
        """
        metrics = {}
        for label, class_name in self.id2label.items():
            metrics[f"{class_name}_acc{suffix}"] = accuracy_score(
                labels[labels == label], preds[labels == label]
            )
        return metrics

    def on_validation_epoch_end(self) -> None:
        """
        Computes validation metrics.
        """

        # Convert torch.tensor to np.array
        probs = torch.cat(
            [
                F.softmax(output["logit"].float(), dim=1)
                for output in self.val_outputs
            ],
            dim=0,
        ).cpu().numpy()
        probs = probs / probs.sum(axis=1)[:, np.newaxis]

        labels = torch.cat(
            [output["label"] for output in self.val_outputs], dim=0
        ).cpu().numpy()

        is_tmas = torch.cat(
            [output["is_tma"] for output in self.val_outputs], dim=0
        ).cpu().numpy()
        
        # Evolutionary algorithm (Add random noise until to find best balanced_acc)
        all_preds = probs.cpu()
        weights = torch.tensor([0.2]*5).cpu()
        best_weights = torch.tensor([0.2]*5).cpu()
        best_score = 0.0
        for _ in range(250):
            for i in range(100):
                if i == 0:
                    noise = weights
                    tmp_preds = all_preds * noise
                else:
                    noise = weights + (torch.rand(5) - 0.5) * 0.05
                    tmp_preds = all_preds * noise
                
                score = balanced_accuracy_score(np.argmax(labels[~is_tmas], axis=1), np.argmax(tmp_preds[~is_tmas], axis=1))
                if score > best_score:
                    best_score = score
                    best_weights = noise
            weights = best_weights
        val_balanced_acc = best_score
        preds = probs.argmax(axis=1)
        labels = labels.numpy().argmax(axis=1)

        # Other Metrics
        val_accuracy = accuracy_score(labels[~is_tmas], preds[~is_tmas])
        val_class_acc = self._accuracy_for_each_class(labels[~is_tmas], preds[~is_tmas])
        val_tma_acc = accuracy_score(labels[is_tmas], preds[is_tmas])

        # Log Metrics
        self.log("val_acc", val_accuracy, prog_bar=True, sync_dist=True)
        self.log("val_tma_acc", val_tma_acc, prog_bar=True, sync_dist=True)
        self.log("val_balanced_acc", val_balanced_acc, prog_bar=True, sync_dist=True)
        self.log_dict(val_class_acc, sync_dist=True)        
        self.val_outputs.clear()
        return
    
    def on_train_epoch_start(self) -> None:
        """
        SWA (Stochastic Weight Averaging) workaround.

        Source:  https://github.com/Lightning-AI/pytorch-lightning/issues/17245
        """
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False

    def on_train_epoch_end(self) -> None:
        """
        Saves model weights at the 12th epoch.
        """
        if self.hparams.save_weights and self.current_epoch == 12:
            save_fpath = "models/{}_{}_{}_{}_{}.pt".format(
                self.hparams.model_name, 
                self.hparams.val_fold, 
                self.hparams.tile_size,
                self.current_epoch,
                self.model_id,
                )
            torch.save(self.model.state_dict(), os.path.join(self.hparams.data_dir, save_fpath))


    def on_train_end(self) -> None:
        """
        Saves model weights at the end of training.
        """
        if self.hparams.save_weights:
            save_fpath = "models/{}_{}_{}_{}_{}.pt".format(
                self.hparams.model_name, 
                self.hparams.val_fold, 
                self.hparams.tile_size,
                self.hparams.epochs,
                self.model_id,
                )
            torch.save(self.model.state_dict(), os.path.join(self.hparams.data_dir, save_fpath))
        return