from turtle import hideturtle
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning import seed_everything
from pathlib import Path


from src.transformer.transformer_utils import *
from src.transformer.metrics import  CorrectedBAcc, CorrectedF1, CorrectedMCC, AUL
import torchmetrics

HOME_PATH = str(Path.home())

class FFN(pl.LightningModule):
    """Hierarchical Attention Network (RNN baseline model)"""
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams.update(hparams)

        # 2. HAN ENCODER
        self.init_encoder()
        # 3. LOSS
        self.init_loss()

        
        # 5. METRICS
        self.init_metrics()

        self.train_step_targets = []
        self.train_step_predictions = []
        self.last_update = 0
        self.last_global_step = 0

    def init_loss(self):
        if self.hparams.loss_type == "robust":
            raise NotImplementedError("Deprecated: use asymmetric loss instead")
        elif self.hparams.loss_type == "asymmetric":
            if self.hparams.encoder_type == "neural":
                self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight)
            elif self.hparams.encoder_type == "logistic":
                self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight, sigmoid=True)
            elif self.hparams.encoder_type == "table":
                self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight, sigmoid=True)

        elif self.hparams.loss_type == "asymmetric_dynamic":
            raise NotImplementedError
        elif self.hparams.loss_type == "entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplemented

    def init_encoder(self):
        if self.hparams.encoder_type == "neural":
            log.info("Encoder is a NEURAL NETWORK")
            self.encoder = NNEncoder(self.hparams)
        elif self.hparams.encoder_type == "logistic":
            log.info("Encoder is a LOGISTIC REGRESSOR")
            self.encoder = LogisticRegression(self.hparams)
        elif self.hparams.encoder_type == "table":
            self.encoder = LifeTable(self.hparams)
        else: 
            raise NotImplementedError()

    def init_metrics(self):
        """Initialise variables to store metrics"""
        ### TRAIN
        self.train_accuracy = torchmetrics.Accuracy(threshold=0.5, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_precision = torchmetrics.Precision(threshold=0.5, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_recall = torchmetrics.Recall(threshold=0.5, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_f1 = torchmetrics.F1Score(threshold=0.5, num_classes=self.hparams.cls_num_targets, average="macro")
        
        ##### VALIDATION
        self.val_cr_bacc = CorrectedBAcc(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.val_cr_f1 = CorrectedF1(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.val_cr_mcc = CorrectedMCC(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.val_aul = AUL()

        ##### TEST
        self.test_cr_bacc = CorrectedBAcc(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.test_cr_f1 = CorrectedF1(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.test_cr_mcc = CorrectedMCC(alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        self.test_aul = AUL()


    def transform_targets(self, targets):
        """Transform Tensor of targets based on the type of loss"""
        if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
            targets = F.one_hot(targets.long(), num_classes = self.hparams.cls_num_targets) ## for custom cross entropy we need to encode targets into one hot representation
        elif self.hparams.loss_type == "entropy":
           targets = targets.long()
        else:
            raise NotImplemented
        return targets


    def forward(self, batch):
        '''Forward pass'''

        return self.encoder(batch["input_ids"]) # output: [bs, hs]

    def train_epoch_start(self, *args):
        """"""
        self.last_global_step = self.global_step
        seed_everything(self.hparams.seed + self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):
        '''Training Iteration'''
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"])
        loss = self.loss(predicted, target=targets)
        ## 3. METRICS
        self.train_step_predictions.append(predicted.detach())
        self.train_step_targets.append(targets.detach())

        if ((self.global_step + 1) % (self.trainer.log_every_n_steps) == 0) and (
            self.last_update != self.global_step
        ):
            self.last_update = self.global_step
            self.log_metrics(
                predictions=torch.cat(self.train_step_predictions),
                targets=torch.cat(self.train_step_targets),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
            del self.train_step_predictions
            del self.train_step_targets
            self.train_step_targets = []
            self.train_step_predictions = []

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Step"""

        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"])
        loss = self.loss(predicted, target=targets)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicted.detach(),
            targets=targets.detach(),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
        )

        return #F.softmax(predicted, dim=1), targets, predicted

    def test_step(self, batch, batch_idx):
        """Test and collect stats"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"])
        loss = self.loss(predicted, target=targets)


        ## 3. METRICS
        self.log_metrics(predictions = predicted.detach(), 
                         targets = targets.detach(), 
                         loss = loss.detach(), 
                         stage = "test", 
                         on_step = False, on_epoch = True)

        return predicted.detach(), targets.detach()

    def test_epoch_end(self, outputs):
        """On Test Epoch End"""
        pass
        #self.calculate_aul(outputs=outputs, stage="test")
        #predictions, targets = [], []
        #for preds, targs in outputs:
        #    predictions.append(preds)
        #    targets.append(targs)

        #predictions = F.softmax(torch.cat(predictions), dim = 1)
        #targets = torch.cat(targets)

        #if targets.shape[-1] == 2: #if targets are one-hot encoded we need to return them back into original 1D 
        #    targets = targets[:,1]     

    
        #print("[TEST END] NUMBER OF POS TARGETS: %s (TOTAL: %s)" %(torch.sum(targets).cpu().detach().numpy(), targets.shape[0]))

    def configure_optimizers(self):
        """"""
        optimizer = torch.optim.RAdam(self.parameters(),
                                     lr =  self.hparams.learning_rate,
                                     weight_decay = self.hparams.weight_decay,
                                     betas=(self.hparams.beta1, self.hparams.beta2),
                                     eps=self.hparams.epsilon)     

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=0.9
                ), 
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        }
    def log_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True):             
        """Compute on step/epoch metrics"""
        assert stage in ["train", "val", "test"]
        if self.hparams.encoder_type == "neural":
            scores = F.softmax(predictions, dim=1)
        else:
            predictions = torch.sigmoid(predictions)
            scores = torch.zeros((predictions.size(0),2), dtype=predictions.dtype, device= predictions.device)
            scores[:,0] += torch.ones(predictions.size(0), dtype=predictions.dtype, device= predictions.device)
            scores[:,0] -= predictions.squeeze()
            scores[:,1] += predictions.squeeze()

        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("train/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("train/pos_predictions", sum(scores[:,0]<0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
            else:
                self.log("train/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)

            self.log("train/accuracy", self.train_accuracy(scores, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/recall", self.train_recall(scores, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/precision", self.train_precision(scores, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/f1", self.train_f1(scores, targets), on_step=on_step, on_epoch = on_epoch)

        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("val/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("val/pos_predictions", sum(scores[:,0]<0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
            else:
                self.log("val/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   
            
            self.val_cr_f1.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.val_cr_bacc.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.val_cr_mcc.update(scores[:,1], targets[:,1])  # this should not happen in self.log 
            self.val_aul.update(scores[:,1], targets[:,1]) # this should not happen in self.log 

            self.log("val/f1_corrected", self.val_cr_f1, on_step = False, on_epoch = True)
            self.log("val/bacc_corrected", self.val_cr_bacc, on_step = False, on_epoch = True)
            self.log("val/mcc_corrected", self.val_cr_mcc, on_step = False, on_epoch = True)
            self.log("val/aul", self.val_aul, on_step = False, on_epoch = True)

        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("test/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("test/pos_predictions", sum(scores[:,0]<0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)

            else:
                self.log("test/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   

            self.test_cr_f1.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.test_cr_bacc.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.test_cr_mcc.update(scores[:,1], targets[:,1])  # this should not happen in self.log 
            self.test_aul.update(scores[:,1], targets[:,1]) # this should not happen in self.log 

            self.test_trg.update(targets[:,1])
            self.test_prb.update(scores[:,1])


            self.log("test/f1_corrected", self.test_cr_f1, on_step=False, on_epoch = True)
            self.log("test/bacc_corrected", self.test_cr_bacc, on_step=False, on_epoch = True)
            self.log("test/mcc_corrected", self.test_cr_mcc, on_step=False, on_epoch = True)
            self.log("test/aul", self.test_aul, on_step = False, on_epoch = True)

    def on_test_start(self, **kwargs):
        self.test_trg = torchmetrics.CatMetric()
        self.test_prb = torchmetrics.CatMetric()

class NNEncoder(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.Linear(hparams.input_size, hparams.hidden_size ), 
                                  Swish(),
                                  ScaleNorm(hidden_size=hparams.hidden_size),
                                  nn.AlphaDropout(p=hparams.dropout))
        self.ff2 = nn.Sequential(nn.Linear(hparams.hidden_size, hparams.hidden_size * 4 ), 
                                  Swish(),
                                  ScaleNorm(hidden_size=hparams.hidden_size * 4),
                                  nn.AlphaDropout(p=hparams.dropout))
        self.ff3 = nn.Sequential(nn.Linear(hparams.hidden_size * 4, hparams.hidden_size * 2 ), 
                                  Swish(),
                                  ScaleNorm(hidden_size=hparams.hidden_size * 2),
                                  nn.AlphaDropout(p=hparams.dropout / 3))

        module = [nn.Sequential(nn.Linear(hparams.input_size, hparams.hidden_size), 
                                  Swish(),
                                  ScaleNorm(hidden_size=hparams.hidden_size),
                                  nn.AlphaDropout(p=hparams.dropout))]
        for i in range(hparams.n_layers-1):
            module.append(nn.Sequential(nn.Linear(hparams.hidden_size, hparams.hidden_size), 
                                  Swish(),
                                  ScaleNorm(hidden_size=hparams.hidden_size),
                                  nn.AlphaDropout(p=hparams.dropout)))

        print("NUM Modules:", len(module))

        self.ff = nn.ModuleList(module)

        self.out = nn.Linear(hparams.hidden_size, hparams.cls_num_targets)

    def forward(self, x):
        """Forward pass"""
        for ff in self.ff:
            x = ff(x)
        return self.out(x)



class LogisticRegression(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.ff = nn.Linear(hparams.input_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.ff(x)
        #return self.ff(x)

class LifeTable(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.ff = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.ff(x[:, 3:5])