from pytorch_lightning import LightningModule
import torch

import time
from torch.nn.functional import cross_entropy

import logging
log = logging.getLogger(__name__)

class DummyModel(LightningModule):

    automatic_optimization = False

    def __init__(self):
        super().__init__()
        self.deserves_gradient = torch.nn.Parameter(torch.tensor(0.))
        self.elapsed_training_steps = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.elapsed_epochs = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)

    def training_step(self, batch, batch_idx):

        assert len(batch.shape) == 1

        for i, x in enumerate(batch):
            self.log(f"batch_{i}", x.to(torch.float))

        self.elapsed_training_steps += 1

        self.log("elapsed_training_steps", self.elapsed_training_steps)
        self.log("elapsed_epochs", self.elapsed_epochs)
        self.log("random_value", torch.randn(size=tuple()))

        return None

    def on_train_epoch_end(self) -> None:

        self.elapsed_epochs += 1
        return None

    def validation_step(self, batch, batch_idx):

        
        return None

    def configure_optimizers(self):
        pass
