from abc import ABC
from base64 import decode
from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import os
from pathlib import Path
import logging

"""Custom code"""
from src.transformer.transformer_utils import *
from src.transformer.transformer import Transformer, MaskedLanguageModel, CLS_Decoder

log = logging.getLogger(__name__)
HOME_PATH = str(Path.home())

class TransformerEncoder(pl.LightningModule):
    """Transformer with Masked Language Model"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        self.hparams.update(hparams)
        #self.idx2token = self.load_lookup(HOME_PATH + self.hparams.dict_path)
        self.last_global_step = 0
        # 1. ENCODER
        self.transformer = Transformer(self.hparams)

        # 2. DECODER BLOCK
        self.task = self.hparams.training_task
        log.info("Training task: %s" %self.task)
        if "mlm" in self.task:
            self.register_buffer("cls_w", torch.tensor(0.2))
            self.register_buffer("mlm_w", torch.tensor(0.8))
            self.register_buffer("cls_a", torch.tensor([1/0.9, 1/0.1, 1/0.1]))
            self.num_outputs = self.hparams.vocab_size
            ### 2.1. DECODERS
            self.mlm_decoder = MaskedLanguageModel(self.hparams, self.transformer.embedding, act="tanh")
            self.cls_decoder = CLS_Decoder(self.hparams)
            ## 2.2. LOSS
            self.cls_loss = nn.CrossEntropyLoss(weight=self.cls_a, label_smoothing=0.1)
            self.mlm_loss = nn.CrossEntropyLoss(ignore_index = 0)
        else:
            raise NotImplementedError()

        # 3. METRICS
        self.init_metrics()

    def init_metrics(self):
        ##### TRAIN
        top_k = 5 if self.num_outputs == self.hparams.vocab_size else 1

        self.train_accuracy = torchmetrics.Accuracy(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_precision = torchmetrics.Precision(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_recall = torchmetrics.Recall(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_f1 = torchmetrics.F1Score(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_cls_acc = torchmetrics.Accuracy(
            threshold=0.5,
            num_classes=3,
            average="macro"
        )

        self.train_cls_f1 = torchmetrics.F1Score(
            threshold=0.5,
            num_classes=3,
            average="macro",
        )

        ##### VALIDATION
        self.val_accuracy = torchmetrics.Accuracy(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_precision = torchmetrics.Precision(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_recall = torchmetrics.Recall(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_f1 = torchmetrics.F1Score(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            mdmc_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_cls_acc = torchmetrics.Accuracy(
            threshold=0.5,
            num_classes=3,
            average="macro"
        )

        self.val_cls_f1 = torchmetrics.F1Score(
            threshold=0.5,
            num_classes=3,
            average="macro",
        )
        
    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        ## 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        ## 3. CLS TASK
        cls_pred  = self.cls_decoder(predicted[:,0])
        return mlm_pred, cls_pred

    def training_step(self, batch, batch_idx):
        """Training Iteration"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = self.cls_loss(cls_preds, target = cls_targs)

        self.log("train/loss_mlm", mlm_loss.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_cls", cls_loss.detach(), on_step=True, on_epoch=True)

        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss
        ## 3. METRICS
        if (self.global_step + 1) % (self.trainer.log_every_n_steps) == 0:
            self.log_metrics(
                predictions=(mlm_preds.detach(), cls_preds.detach()),
                targets=(mlm_targs.detach(),  cls_targs.detach()),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
        return loss

    def train_epoch_start(self, *args):
        """On Epoch Start"""
        self.last_global_step = self.global_step
        seed_everything(self.hparams.seed + self.trainer.current_epoch)

    def training_epoch_end(self, output):
        """On Epoch End"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)

    def validation_epoch_end(self, outputs) -> None:
        """Save the embedding on validation epoch end"""
        return super().validation_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        """Validation Step"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = self.cls_loss(cls_preds, target = cls_targs)

        self.log("val/loss_mlm", mlm_loss.detach(), on_step=True, on_epoch=True)
        self.log("val/loss_cls", cls_loss.detach(), on_step=True, on_epoch=True)

        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss

        ## 3. METRICS
        self.log_metrics(
            predictions=(mlm_preds.detach(), cls_preds.detach()),
            targets=(mlm_targs.detach(),  cls_targs.detach()),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = self.cls_loss(cls_preds, target = cls_targs)

        self.log("val/loss_mlm", mlm_loss.detach(), on_step=False, on_epoch=True)
        self.log("val/loss_cls", cls_loss.detach(), on_step=False, on_epoch=True)

        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss
        ## 3. METRICS
        self.log_metrics(
            predictions=(mlm_preds.detach(), cls_preds.detach()),
            targets=(mlm_targs.detach(),  cls_targs.detach()),
            loss=loss.detach(),
            stage="test",
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        """"""
        no_decay = [
            "bias",
            "norm",
            "age",
            "abspos",
            "token",
            "decoder.g"
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate, 
                    epochs=30, steps_per_epoch=375,
                    three_phase=False, pct_start=0.05, max_momentum=self.hparams.beta1,
                    div_factor=30
                ),  
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step: bool = True,
        on_epoch: bool = True,
    ):
        """Compute on step/epoch metrics"""
        loss = loss.cpu()
        cls_preds = F.softmax(predictions[1], dim=-1)
        mlm_preds = F.softmax(predictions[0], dim=-1).permute(0, 2, 1)

        cls_targs = targets[1]
        mlm_targs = targets[0]

        if stage == "train":

            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "train/accuracy",
                self.train_accuracy(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall",
                self.train_recall(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision",
                self.train_precision(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f1",
                self.train_f1(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/cls_acc",
                self.train_cls_acc(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/cls_f1",
                self.train_cls_f1(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )


        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "val/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "val/accuracy",
                self.val_accuracy(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/recall",
                self.val_recall(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/precision",
                self.val_precision(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/f1",
                self.val_f1(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/cls_acc",
                self.val_cls_acc(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/cls_f1",
                self.val_cls_f1(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    @staticmethod
    def load_lookup(path):
        """Load Token-to-Index and Index-to-Token dictionary"""
        with open(path, "rb") as f:
            indx2token, token2indx = pickle.load(f)
        return indx2token, token2indx