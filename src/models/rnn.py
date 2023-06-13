from turtle import forward
from ..transformer.cls_model import Transformer_CLS, Transformer_PSY
from ..transformer.transformer import AttentionDecoder, AttentionDecoderP, CLS_Decoder_FT2, Deep_Decoder
from ..transformer.embeddings import Embeddings
import torch
import torch.nn as nn
import logging


log = logging.getLogger(__name__)



class SimpleGRU_PSY(Transformer_PSY):
    def init_encoder(self):
        self.transformer = GRU_Encoder(self.hparams)

    def init_decoder(self):
        num_outputs = self.num_outputs
        if self.hparams.loss_type == "corn":
            num_outputs -= 1
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoderP(self.hparams, num_outputs = num_outputs)
            self.encoder_f = self.transformer.forward
        else: 
            log.info("Model with the CLS representation")
            self.decoder = Deep_Decoder(self.hparams, num_outputs = num_outputs)
            self.encoder_f = self.transformer.forward
    def configure_optimizers(self):
        optimizer_grouped_parameters = list()
        no_decay = [ 
           "bias",
           "norm"
        ]

        optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not (nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.learning_rate,
                }
            )
        optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if  (nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.,
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
                momentum=0.7,
                dampening=0.1)
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
    def training_epoch_end(self, output):
        """On Epoch End"""
        pass

class SimpleGRU(Transformer_CLS):
    def init_encoder(self):
        self.transformer = GRU_Encoder(self.hparams)

    def init_decoder(self):
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_f = self.transformer.forward
        else: 
            log.info("Model with the CLS representation")
            self.decoder = CLS_Decoder_FT2(self.hparams)
            self.encoder_f = self.transformer.forward_cls
    def configure_optimizers(self):
        optimizer_grouped_parameters = list()
        no_decay = [ 
           "bias",
           "norm"
        ]

        optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not (nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.learning_rate,
                }
            )
        optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if  (nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.,
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
                momentum=0.7,
                dampening=0.1)
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
    def training_epoch_end(self, output):
        """On Epoch End"""
        pass


class GRU_Encoder(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams
        # 1.EMBEDDING UNIT
        self.embedding = Embeddings(hparams=hparams)
        # 2. ENCODER BLOCKS
        self.encoder = nn.GRU(input_size = self.hparams.hidden_size, 
                              hidden_size = (self.hparams.hidden_size // 2 if self.hparams.bidirectional == True else self.hparams.hidden_size),
                              bidirectional = self.hparams.bidirectional, 
                              num_layers = self.hparams.n_layers,
                              dropout = (0 if self.hparams.n_layers==1 else self.hparams.fw_dropout),
                              batch_first = True)
    def forward(self, x, padding_mask):
        """Forward pass"""
        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        x, _ = self.encoder(x)
        x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
        return x

    def forward_cls(self, x, padding_mask):
        x = self.forward(x, padding_mask)
        return x[:,0]

