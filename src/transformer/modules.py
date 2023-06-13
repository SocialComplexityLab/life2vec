import torch
import torch.nn as nn
import math
import time

"""Custom code"""
from src.transformer.attention import MultiHeadAttention
from src.transformer.transformer_utils import *

import logging

log = logging.getLogger(__name__)


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class EncoderLayer(nn.Module):
    """Encoder Block"""

    def __init__(self, hparams):
        """"""
        super(EncoderLayer, self).__init__()

        assert (
            hparams.hidden_size % hparams.n_heads == 0
        ), "Encoder: Incorrect hidden_size (%s, %s)" % (
            hparams.hidden_size,
            hparams.n_heads,
        )
        start = time.time()

        self.attention = MultiHeadAttention(hparams)
        self.attention_sublayer = AttentionConnection(hidden_size=hparams.hidden_size, norm_type=hparams.norm_type)

        self.position_wise = PositionWiseFeedForward(hparams)
        self.position_sublayer = SublayerConnection(hidden_size=hparams.hidden_size)

        log.info("EncoderLayer setup is finised:  %.3f s" % (time.time() - start))

    def redraw_projection_matrix(self):
        """Redraw projection matrices during the training"""
        try:
            try:
                self.attention.attention.fast_attention.redraw_projection_matrix("cuda")
            except:
                self.attention.attention.fast_attention.redraw_projection_matrix("cpu")
        except:
            print("ENCODER: Cannot redraw a projection. Wrong attention type")

    def forward(self, x, mask=None, pos=None):
        """Forward Pass"""
        x = self.attention_sublayer(x, mask, pos, sublayer=self.attention)

        x = self.position_sublayer(x, sublayer=self.position_wise)

        return x
