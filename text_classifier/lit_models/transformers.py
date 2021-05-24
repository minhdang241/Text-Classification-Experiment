import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb

from .base import BaseLitModel


class TransformerLitModel(BaseLitModel):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.

    The module must take x, y as inputs, and have a special predict() method.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.model = model
        self.mapping = self.model.data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.model(batch)
        loss = self.loss_fn(outputs.logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.model(batch)
        loss = self.loss_fn(outputs.logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(F.softmax(outputs.logits, dim=-1), labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.model(batch)
        self.test_acc(F.softmax(outputs.logits, dim=-1), labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
