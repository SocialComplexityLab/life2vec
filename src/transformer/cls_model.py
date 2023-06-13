
from cProfile import label
from dataclasses import replace
from src.callbacks import HOME_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics 
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from focal_loss.focal_loss import FocalLoss
from imblearn.metrics import macro_averaged_mean_absolute_error
from scipy.stats import median_abs_deviation
from sklearn.metrics import f1_score, cohen_kappa_score
import logging

"""Custom code"""
from src.transformer.transformer_utils import *
from src.transformer.transformer import  AttentionDecoder, CLS_Decoder, AttentionDecoderP, Deep_Decoder, Transformer
from src.transformer.metrics import  CorrectedBAcc, CorrectedF1, CorrectedMCC, AUL
from coral_pytorch.losses import corn_loss
from pathlib import Path
import os

HOME_PATH = str(Path.home())
log = logging.getLogger(__name__)
REG_LOSS = ["mae", "mse", "smooth"]
CLS_LOSS = ["entropy", "focal", "ordinal", "corn", "cdw", "nll_loss"]

class Transformer_CLS(pl.LightningModule):
    """Transformer with Classification Layer"""

    def __init__(self, hparams):
        super(Transformer_CLS, self).__init__()
        self.hparams.update(hparams)


        # 1. Transformer: Load pretrained encoders
        self.init_encoder()
        # 2. Decoder
        self.init_decoder()
        # 3. Loss
        self.init_loss()
        # 4. Metric
        self.init_metrics()
        self.init_collector()
        # 5. Embeddings
        self.embedding_grad()
        # x. Logging params
        self.train_step_targets = []
        self.train_step_predictions = []
        self.last_update = 0
        self.last_global_step = 0

    @property
    def num_outputs(self):
        return self.hparams.num_targets

    def init_encoder(self):
        self.transformer = Transformer(self.hparams)
        log.info("Embedding sample before load: %.2f" %self.transformer.embedding.token.weight[1, 0].detach())
        if "none" in self.hparams.pretrained_model_path:
            log.info("No pretrained model")
        else:
            log.info("Pretrained Model Path:\n\t%s" %(HOME_PATH + self.hparams.pretrained_model_path))
            self.transformer.load_state_dict(
                torch.load(HOME_PATH + self.hparams.pretrained_model_path, map_location=self.device), strict=False
            )
        log.info("Embedding sample after load: %.2f" %self.transformer.embedding.token.weight[1, 0].detach())

    def init_decoder(self):
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_f = self.transformer.forward_finetuning
        else: 
            log.info("Model with the CLS representation")
            self.decoder = CLS_Decoder(self.hparams)
            self.encoder_f = self.transformer.forward_finetuning_cls

    def init_loss(self):
    
        if self.hparams.loss_type == "robust":
            raise NotImplementedError("Deprecated: use asymmetric loss instead")
        elif self.hparams.loss_type == "asymmetric":
            self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight)
        elif self.hparams.loss_type == "asymmetric_dynamic":
            raise NotImplementedError
        elif self.hparams.loss_type == "entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplemented

    def embedding_grad(self):
        ### Remove parameters from the computational graph
        if self.hparams.freeze_positions:
            for param in self.transformer.embedding.age.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.abspos.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.segment.parameters():
                param.requires_grad = False

    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.encoder_f(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long()
        )
        ## 2. CLS Predictions 
        if self.hparams.pooled: 
            predicted = self.decoder(predicted, mask = batch["padding_mask"].long())
        else:
            predicted = self.decoder(predicted)

        return predicted 

    
    def forward_with_embeddings(self, embeddings, meta = None):
        """Same as forward, but takes pre-calculated embeddings instead of tokens"""
        ## 2. CLS Predictions 
        if self.hparams.pooled:
            predicted = self.transformer.forward_finetuning_with_embeddings(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted, meta["padding_mask"].long())
        
        else:
            predicted = self.transformer.forward_finetuning_with_embeddings_cls(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted)
        return predicted         

    def init_metrics(self):
        """Initialise variables to store metrics"""
        ### TRAIN
        self.train_accuracy = torchmetrics.Accuracy(threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_precision = torchmetrics.Precision(threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_recall = torchmetrics.Recall(threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_f1 = torchmetrics.F1Score(threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        
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

    def init_collector(self):
        """Collect predictions and targets"""
        self.test_trg = torchmetrics.CatMetric()
        self.test_prb = torchmetrics.CatMetric()
        self.test_id  = torchmetrics.CatMetric()
        self.val_trg = torchmetrics.CatMetric()
        self.val_prb = torchmetrics.CatMetric()
        self.val_id  = torchmetrics.CatMetric()

    def on_train_epoch_start(self, *args):
        """"""
        seed_everything(self.hparams.seed + self.trainer.current_epoch)
    #    log.info("Redraw Projection Matrices")

    def on_train_epoch_end(self, *args):
        if self.hparams.attention_type == "performer":
            log.info("Redraw Projection Matrices")
            self.transformer.redraw_projection_matrix(-1)


    def transform_targets(self, targets, seq, stage: str):
        """Transform Tensor of targets based on the type of loss"""
        if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
            targets = F.one_hot(targets.long(), num_classes = self.hparams.num_targets) ## for custom cross entropy we need to encode targets into one hot representation
        elif self.hparams.loss_type == "entropy":
           targets = targets.long()
        else:
            raise NotImplemented
        return targets

    def training_step(self, batch, batch_idx):
        """Training Iteration"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="train")

        loss = self.loss(predicted, targets)
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

        ## 5. RETURN LOSS
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Step"""

        ## 1. ENCODER-DECODER
        predicted = self(batch)
        predicted = predicted
        ## 2. LOSS
        targets = self.transform_targets(batch["target"], seq = batch["input_ids"], stage="val")
        loss = self.loss(predicted, targets)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicted.detach(),
            targets=targets.detach(),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
            sid = batch["sequence_id"]
        )

        return None

    def test_step(self, batch, batch_idx):
        """Test and collect stats"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="test")
        loss = self.loss(predicted, targets)
        ## 3. METRICS
        self.log_metrics(predictions = predicted.detach(), targets = targets.detach(), loss = loss.detach(), sid = batch["sequence_id"], stage = "test", on_step = False, on_epoch = True)
        return None

    def on_after_backward(self) -> None:
        #### Freeze Embedding Layer, except for the CLS token
        if self.hparams.parametrize_emb:
             w = self.transformer.embedding.token.parametrizations.weight.original
        else:
            w = self.transformer.embedding.token.weight
        if self.hparams.freeze_embeddings:
            w.grad[0] = 0
            w.grad[2:8] = 0
            w.grad[10:] = 0
        return super().on_after_backward()


    def debug_optimizer(self):
        print("==========================")
        print("PARAMETERS IN OPTIMIZER")
        print("\tGROUP 1:")
        for n, p in self.named_parameters():
            if "embedding" in n:
                print("\t\t", n)
        print("\tGROUP 2:")
        for i in range(0, self.hparams.n_encoders):
            for n, p in self.named_parameters():
                if "encoders.%s." % i in n:
                    print("\t\t", n)
        print("\tGROUP 3:")
        for n, p in self.named_parameters():
            if "decoder" in n:
                print("\t\t", n)
   
    def configure_optimizers(self):
        """"""
        self.debug_optimizer()
        no_decay = [ 
           "bias",
           "norm"
        ]
        optimizer_grouped_parameters = list()
        lr = self.hparams.learning_rate * (self.hparams.layer_lr_decay**self.hparams.n_encoders)
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if ("embedding" in n) 
                ],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )
        for i in range(0, self.hparams.n_encoders):
            lr = self.hparams.learning_rate * (
                (self.hparams.layer_lr_decay) ** (self.hparams.n_encoders - i)
            )  # lwoer layers should have lower learning rate

            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and not (nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": lr,
                }
            )
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and  (nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in self.named_parameters() if "decoder" in n],
                "weight_decay": self.hparams.weight_decay_dc,
                "lr": self.hparams.learning_rate,
            }
        )

        if self.hparams.optimizer_type == "radam":
            optimizer = torch.optim.RAdam(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                momentum=0.0,
                dampening=0.00)
        elif self.hparams.optimizer_type == "asgd":
            optimizer = torch.optim.ASGD(
                optimizer_grouped_parameters,
                t0=3000)
        elif self.hparams.optimizer_type == "adamax":
            optimizer = torch.optim.Adamax(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        # no sgd , adsgd, nadam, rmsprop
        else:
            raise NotImplementedError

        if self.hparams.stage in ["search", "finetuning"]:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.hparams.lr_gamma
                ), 
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        }


    def log_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid = None):             
        """Compute on step/epoch metrics"""
        assert stage in ["train", "val", "test"]
        scores = F.softmax(predictions, dim=1)
        
        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("train/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("train/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
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
                self.log("val/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
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
            self.val_trg.update(targets[:,1])
            self.val_prb.update(scores[:,1])
            self.val_id.update(sid)

        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("test/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("test/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)

            else:
                self.log("test/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   

            self.test_cr_f1.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.test_cr_bacc.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.test_cr_mcc.update(scores[:,1], targets[:,1])  # this should not happen in self.log 
            self.test_aul.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            self.test_trg.update(targets[:,1])
            self.test_prb.update(scores[:,1])
            self.test_id.update(sid)

            self.log("test/f1_corrected", self.test_cr_f1, on_step=False, on_epoch = True)
            self.log("test/bacc_corrected", self.test_cr_bacc, on_step=False, on_epoch = True)
            self.log("test/mcc_corrected", self.test_cr_mcc, on_step=False, on_epoch = True)
            self.log("test/aul", self.test_aul, on_step = False, on_epoch = True)
    

    @staticmethod
    def load_lookup(path):
        """Load Token-to-Index and Index-to-Token dictionary"""
        with open(path, "rb") as f:
            indx2token, token2indx = pickle.load(f)
        return indx2token, token2indx


###################################################################################
### PSY MODEL #####################################################################
###################################################################################

class Transformer_PSY(Transformer_CLS):

    def __init__(self, hparams):
        super().__init__(hparams)
        #self.automatic_optimization = False
        self.step_data = list()
        self.mom_last_loss = 10.
        self.mom_current_loss = 0.
        self.train_step_loss = list()
        self.MEDIAN_OVER_N = 8
        self.ACCUMULATE_GRADIENT_OVER_N = 2
        self.batch_counter = 0

    @property
    def num_outputs(self):
        if self.hparams.loss_type in REG_LOSS:
            return 1
        else:
            return self.hparams.num_classes

    def init_decoder(self):
        num_outputs = self.num_outputs
        if self.hparams.loss_type == "corn":
            num_outputs -= 1
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoderP(self.hparams, num_outputs = num_outputs)
            self.encoder_f = self.transformer.forward_finetuning
        else: 
            log.info("Model with the CLS representation")
            self.decoder = Deep_Decoder(self.hparams, num_outputs = num_outputs)#MosDecoder(self.hparams, num_outputs=self.num_outputs, k=3)
            self.encoder_f = self.transformer.forward_finetuning

    def init_loss(self):
        print("LOSS TYPE:", self.hparams.loss_type)
        if self.hparams.weighted_loss:
            reduction = "none"
        else:
            reduction = "mean"
        if self.hparams.loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.hparams.loss_type == "entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.hparams.loss_type == "ordinal":
            self.loss_fn = CumulativeLinkLoss()
        elif self.hparams.loss_type == "nll_loss":
            self.loss_fn = nn.NLLLoss()
        elif self.hparams.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.hparams.loss_type == "smooth":
            self.loss_fn = nn.SmoothL1Loss(beta=0.2, reduction=reduction)#, reduce=False)
        elif self.hparams.loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=2.)
        elif self.hparams.loss_type == "corn":
            self.loss_fn = lambda x, y: corn_loss(x, y, num_classes = self.num_outputs)
        elif self.hparams.loss_type =="cdw":
            self.loss_fn = CDW_CELoss(num_classes=self.num_outputs, alpha=2)
        else:
            raise Exception("Wrong Loss Types")
        self.base_loss_fn = FocalLoss(gamma=2.) #nn.CrossEntropyLoss(label_smoothing=0.3) #FocalLoss(gamma=2.)#nn.CrossEntropyLoss()
        self.smooth_fn = nn.CrossEntropyLoss(label_smoothing=1.)


    def train_forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.encoder_f(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long()
        )

        predicted_mom = self.encoder_mom_f(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long()
        )
        ## 2. CLS Predictions 
        predicted = self.decoder(predicted)
        predicted_mom = self.decoder_mom(predicted_mom)

        return predicted, predicted_mom

    def sub_training_step_A(self, batch, targets, batch_idx):
        opt = self.optimizers()
        for param in self.transformer_mom.parameters():
            param.requires_grad_(False)
        for param in self.decoder_mom.parameters():
            param.requires_grad_(False)

        predicted, _ = self.train_forward(batch)
        loss = self.loss(predicted, targets)
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val =1., gradient_clip_algorithm="value")
        opt.step()
        opt.zero_grad()

        for param in self.transformer_mom.parameters():
            param.requires_grad_(True)
        for param in self.decoder_mom.parameters():
            param.requires_grad_(True)

        self.train_step_predictions.append(predicted.detach())
        self.train_step_targets.append(targets.detach())

        if self.last_update != self.global_step:
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
            torch.cuda.empty_cache()
        return loss

    def loss(self, preds, targs):
        if self.hparams.loss_type in REG_LOSS:
            return self._loss_regression(preds,targs)
        elif self.hparams.loss_type in CLS_LOSS:
            weight = torch.tensor([[0.62, 0.19, 0.09, 0.05, 0.04],
                                   [0.39, 0.23, 0.18, 0.13, 0.12],
                                   [0.39, 0.23, 0.15, 0.14, 0.10],
                                   [0.59, 0.22, 0.09, 0.05, 0.05]] ,device = self.device)
            ### multiprediction
            num_classes = 4
            output = torch.sum(
                torch.hstack(
                [self._loss_classification(preds[:,i], targs[:,i], weight=weight[i]) for i in range(num_classes)]))

            return output
        else:
            raise ValueError("wrong loss type")
       
    def _loss_classification(self, preds, targs, weight=None):
        preds_ = self.sigsoftmax(preds)
        if self.hparams.loss_type == "focal":
            preds = torch.softmax(preds, -1)
        elif self.hparams.loss_type == "nll_loss":
            return self.loss_fn(F.log_softmax(preds, dim=-1), targs)
        elif self.hparams.loss_type == "cdw":
            return   .5 * self.loss_fn(preds_, targs.float()) \
                    + 1. * self.base_loss_fn(preds_,targs.long()) \
                    + 0.2 * self.smooth_fn(preds, targs.long())
        return self.loss_fn(preds, targs)
    def _loss_regression(self, preds, targs):
        if not self.hparams.weighted_loss:
            return self.loss_fn(preds.view(-1),targs[:,0].view(-1))
        if torch.isclose(targs[:,1].sum(),  torch.zeros_like(targs[:,1].sum())):
            return self.loss_fn(preds.view(-1),targs[:,0]).mean()
        return torch.mul(self.loss_fn(preds.view(-1),targs[:,0]),targs[:,1]).mean()

    def transform_targets(self, targets, seq, stage):
        # ['HH','EM','EX','AG','CO','OP', "SDO", "SVOa", "RISK", "CRTi", "CRTr"]
        TRG_ID1, TRG_ID2, TRG_ID3, TRG_ID4 = 19, 22,43, 69
        trg1 = targets[:,TRG_ID1].unsqueeze(-1) - 1
        trg2 = targets[:,TRG_ID2].unsqueeze(-1) - 1
        trg3 = targets[:,TRG_ID3].unsqueeze(-1) - 1
        trg4 = targets[:,TRG_ID4].unsqueeze(-1) - 1

        return torch.hstack([trg1, trg2, trg3,trg4])

     

    def on_train_epoch_start(self, **args):
        """"""
        try:
            self.train_mae.reset()
            self.train_mse.reset()
        except:
            pass

    def on_validation_start(self, **args):
        """"""
        try:
            torch.random.manual_seed(1)
            self.val_mae.reset()
            self.val_mse.reset()
        except:
            pass
    #def on_test_start(self, **args):



    def init_metrics(self):
        ### TRAIN
        if self.hparams.loss_type in REG_LOSS:

            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.train_mse = torchmetrics.MeanSquaredError()

            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.val_mse = torchmetrics.MeanSquaredError()

            self.rnd_mae = torchmetrics.MeanAbsoluteError()
            self.rnd_mse = torchmetrics.MeanSquaredError()

        elif self.hparams.loss_type in CLS_LOSS:

            self.sigsoftmax = SigSoftmax()

            self.train_acc = torchmetrics.Accuracy(num_classes=self.num_outputs, average="macro")
            self.train_f1 =  torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")
            self.train_mcc = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs)
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.train_qwk = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")


            self.val_acc = torchmetrics.Accuracy(num_classes=self.num_outputs, average="macro")
            self.val_f1 =  torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")
            self.val_mcc = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs, average="macro")
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.val_qwk = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")

            self.test_mcc_dict, self.test_qwk_dict, self.test_mae_dict, self.test_acc_dict, self.test_rmae_dict = {}, {}, {}, {}, {}
            self.test_mcc_0 = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs, average="macro")
            self.test_qwk_0 = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")
            self.test_mae_0 = torchmetrics.MeanAbsoluteError()
            self.test_f1_0 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")

            self.test_mcc_1 = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs, average="macro")
            self.test_qwk_1 = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")
            self.test_mae_1 = torchmetrics.MeanAbsoluteError()
            self.test_f1_1 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")

            self.test_mcc_2 = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs, average="macro")
            self.test_qwk_2 = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")
            self.test_mae_2 = torchmetrics.MeanAbsoluteError()
            self.test_f1_2 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")

            self.test_mcc_3 = torchmetrics.MatthewsCorrCoef(num_classes=self.num_outputs, average="macro")
            self.test_qwk_3 = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")
            self.test_mae_3 = torchmetrics.MeanAbsoluteError()
            self.test_f1_3 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")


            self.test_mae_r0 = torchmetrics.MeanAbsoluteError()
            self.test_mae_r1 = torchmetrics.MeanAbsoluteError()
            self.test_mae_r2 = torchmetrics.MeanAbsoluteError()
            self.test_mae_r3 = torchmetrics.MeanAbsoluteError()

            self.test_f1_r0 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")
            self.test_f1_r1 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")
            self.test_f1_r2 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")
            self.test_f1_r3 = torchmetrics.F1Score(num_classes=self.num_outputs, average="macro")



            self.test_rmae_dict = {0: self.test_mae_r0,
                                  1: self.test_mae_r1,
                                  2: self.test_mae_r2,
                                  3: self.test_mae_r3,
                                 }

            self.test_rf1_dict = {0: self.test_f1_r0,
                                  1: self.test_f1_r1,
                                  2: self.test_f1_r2,
                                  3: self.test_f1_r3,
                                 }

            self.test_mcc_dict = {0: self.test_mcc_0,
                                  1: self.test_mcc_1,
                                  2: self.test_mcc_2,
                                  3: self.test_mcc_3,
                                 }

            self.test_qwk_dict = {0: self.test_qwk_0,
                                  1: self.test_qwk_1,
                                  2: self.test_qwk_2,
                                  3: self.test_qwk_3,
                                 }
            self.test_mae_dict = {0: self.test_mae_0,
                                  1: self.test_mae_1,
                                  2: self.test_mae_2,
                                  3: self.test_mae_3,
                                 }
            self.test_f1_dict = {0: self.test_f1_0,
                                  1: self.test_f1_1,
                                  2: self.test_f1_2,
                                  3: self.test_f1_3,
                                 }


            self.test_mae = torchmetrics.MeanAbsoluteError()
            self.test_qwk = torchmetrics.CohenKappa(num_classes=self.num_outputs, task="multiclass",  weights="quadratic")


        self.val_trg = torchmetrics.CatMetric()
        self.val_prb = torchmetrics.CatMetric()
        self.test_trgs, self.test_scrs, self.test_rnds = {}, {}, {}
        for i in range(4):
            self.test_trgs[i] = torchmetrics.CatMetric()
            self.test_scrs[i] = torchmetrics.CatMetric()
            self.test_rnds[i] = torchmetrics.CatMetric()

        self.val_trgs, self.val_scrs, self.val_rnds = {}, {}, {}
        for i in range(4):
            self.val_trgs[i] = torchmetrics.CatMetric()
            self.val_scrs[i] = torchmetrics.CatMetric()
            self.val_rnds[i] = torchmetrics.CatMetric()

        self.train_trg = torchmetrics.CatMetric()
        self.train_prb = torchmetrics.CatMetric()
        self.train_prb_full = torchmetrics.CatMetric()
        self.train_ids = torchmetrics.CatMetric()

    def log_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid=None):             
        """Compute on step/epoch metrics"""
        assert stage in ["train", "val", "test"]
        if self.hparams.loss_type in REG_LOSS:
            self._reg_metrics(predictions, targets.unsqueeze(-1), loss, stage, on_step, on_epoch, sid)
        elif self.hparams.loss_type in CLS_LOSS:
            self._cls_metrics(predictions, targets.long(), loss, stage, on_step, on_epoch, sid)


    def  _cls_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid=None): 
        for i in range(4):
            self.local_cls_metrics(i, predictions[:,i], targets[:,i].long(), loss, stage, on_step, on_epoch, sid)

        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)
            self.log("train/mae", self.train_mae(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
            self.log("train/qwk", self.train_qwk(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch = on_epoch)
            self.log("val/mae", self.val_mae(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
            self.log("val/qwk", self.val_qwk(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch = on_epoch)
            self.log("test/mae", self.test_mae(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
            self.log("test/qwk", self.test_qwk(torch.argmax(predictions, -1).view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)


    def local_cls_metrics(self, _id, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid=None): 
        scores = self.sigsoftmax(predictions)
        counts_trgs = list()
        counts_prbs = list()

        _scores = torch.argmax(scores, -1).long()
        for i in range(self.hparams.num_classes):
            counts_trgs.append((targets == i).sum())
            counts_prbs.append((_scores == i).sum())
        print("Sampled trgs (%s):" %_id, torch.stack(counts_trgs))
        print("Sampled prbs (%s):" %_id, torch.stack(counts_prbs))

        if stage == "train":
            self.log("train/acc", self.train_acc(scores, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("train/f1", self.train_f1(scores, targets),on_step=on_step, on_epoch = on_epoch)
            self.train_trg.update(targets)
            self.train_prb.update(scores)

        elif stage == "val":
            self.log("val/acc", self.val_acc(scores, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("val/f1", self.val_f1(scores, targets),on_step=on_step, on_epoch = on_epoch)
            self.val_trg.update(targets)
            self.val_prb.update(scores)
            self.val_trgs[_id].update(targets.view(-1).cpu())
            self.val_scrs[_id].update(_scores.view(-1).cpu())


        elif stage == "test":
            self.log("test/mcc_%s"%_id, self.test_mcc_dict[_id](scores, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("test/qwk_%s"%_id, self.test_qwk_dict[_id](_scores.view(-1,1), targets.view(-1,1)),on_step=on_step, on_epoch = on_epoch)
            rand_score = torch.randint(low = 0, high=5, size=_scores.view(-1,1).shape).to(self.device)
            self.test_trgs[_id].update(targets.view(-1).cpu())
            self.test_scrs[_id].update(_scores.view(-1).cpu())
            self.test_rnds[_id].update(rand_score.view(-1).cpu())
       
    def on_validation_epoch_end(self):
        result = []
        kappas = []
        for i in range(4):
            s = self.val_scrs[i].compute().numpy().astype(int).tolist()
            t = self.val_trgs[i].compute().numpy().astype(int).tolist()
            mamae = macro_averaged_mean_absolute_error(t,s)
            kappa = cohen_kappa_score(t,s, weights="quadratic")
            if self.trainer.current_epoch == 0:
                mamae +=1

            self.log("val/mamae_%s"%i, mamae, on_step=False, on_epoch = True)
            kappas.append(kappa)
            result.append(mamae)
        self.log("val/mamae", np.mean(result), on_step=False, on_epoch = True)
        self.log("val/kappa", np.mean(kappas), on_step=False, on_epoch = True)

    def on_test_epoch_end(self):
        try:
            os.mkdir(HOME_PATH + self.hparams.save_path)
        except:
            pass
        for i in range(4):
            s = self.test_scrs[i].compute().numpy().astype(int)
            t = self.test_trgs[i].compute().numpy().astype(int)
            r = self.test_rnds[i].compute().numpy().astype(int)
            np.save(HOME_PATH + self.hparams.save_path + "/score_%s.npy" %i, s)
            np.save(HOME_PATH +self.hparams.save_path + "/targ_%s.npy" %i, t)


            r_res = list()
            s_res = list()
            rf1_res = list()
            sf1_res = list()
            rkqw_res = list()
            skqw_res = list()
            for _ in range(10000):
                idx = np.random.choice(s.shape[0], size=s.shape[0], replace=True)
                _s = s[idx]
                _t = t[idx]
                _r = r[idx]
                r_res.append(macro_averaged_mean_absolute_error(_t,_r))
                s_res.append(macro_averaged_mean_absolute_error(_t,_s))
                rf1_res.append(f1_score(_t, _r, average = "macro"))
                sf1_res.append(f1_score(_t, _s, average = "macro"))

                rkqw_res.append(cohen_kappa_score(_t, _r, weights= "quadratic"))
                skqw_res.append(cohen_kappa_score(_t, _s, weights= "quadratic"))


            print("=====KQW=====")
            print("SCR ATTR %s:"%i, 
                  "CI [%.3f, %.3f]" %(np.quantile(skqw_res, 0.025), np.quantile(skqw_res, 0.975)))
            print("\tMAD: %.3f" %median_abs_deviation(skqw_res, center=np.mean))
            print("RND ATTR %s:"%i, 
                  "CI [%.3f, %.3f]" %(np.quantile(rkqw_res, 0.025), np.quantile(rkqw_res, 0.975)))
            print("\tMAD: %.3f" %median_abs_deviation(rkqw_res, center=np.mean))

            self.log("test/mamae_%s"%i, macro_averaged_mean_absolute_error(t,s), on_step=False, on_epoch = True)
            self.log("test/rmamae_%s"%i, macro_averaged_mean_absolute_error(t,r), on_step=False, on_epoch = True)

            self.log("test/f1_%s"%i, f1_score(t, s, average = "macro"), on_step=False, on_epoch = True)
            self.log("test/rf1_%s"%i, f1_score(t, r, average = "macro"), on_step=False, on_epoch = True)

            self.log("test/skqw_%s"%i, cohen_kappa_score(t, s, weights= "quadratic"), on_step=False, on_epoch = True)
            self.log("test/rkqw_%s"%i, cohen_kappa_score(t, r, weights= "quadratic"), on_step=False, on_epoch = True)


    def _reg_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid=None):   

        def get_random_preds():
            return torch.normal(mean=0.64, std=0.17, size = predictions.shape, device=predictions.device, dtype=predictions.dtype)


        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)

            self.log("train/mae", self.train_mae(predictions, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("train/mse", self.train_mse(predictions, targets),on_step=on_step, on_epoch = on_epoch)
            self.train_trg.update(targets)
            self.train_prb.update(predictions)

        elif stage == "val":
            predictions = torch.clip(predictions, min = 0., max = 1.)
            predictions_rnd = torch.clip(get_random_preds(), min=0., max=1.)
            self.log("val/loss", loss, on_step=on_step, on_epoch = on_epoch)
            self.log("val/mae", self.val_mae(predictions, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("val/mse", self.val_mse(predictions, targets),on_step=on_step, on_epoch = on_epoch)

            self.log("rnd/mae", self.rnd_mae(predictions_rnd, targets),on_step=on_step, on_epoch = on_epoch)
            self.log("rnd/mse", self.rnd_mse(predictions_rnd, targets),on_step=on_step, on_epoch = on_epoch)

            self.val_trg.update(targets)
            self.val_prb.update(predictions)


        elif stage == "test":
            pass