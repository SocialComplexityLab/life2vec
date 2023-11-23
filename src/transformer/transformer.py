from ast import Num
from re import I
from tkinter import Scrollbar
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

import math
import logging

"""Custom code"""
from src.transformer.embeddings import Embeddings
from src.transformer.transformer_utils import Norm, ReZero, ScaleNorm, SigSoftmax, hard_softmax,  l2_norm, Center, Swish
from src.transformer.transformer_utils import EncoderLayer
from src.models.han import ContextAttention, masked_softmax

log = logging.getLogger(__name__)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "tanh": torch.tanh
}

class Transformer(nn.Module):
    def __init__(self, hparams):
        """Encoder"""
        super(Transformer, self).__init__()

        self.hparams = hparams
        # 1.EMBEDDING UNIT
        self.embedding = Embeddings(hparams=hparams)
        # 2. ENCODER BLOCKS
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )
    def forward(self, x, padding_mask):
        """Forward pass"""
        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        for layer in self.encoders:
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
        return x

    def forward_finetuning(self,x, padding_mask=None):

        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:,3]
        )

        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)

        return x

    def forward_finetuning_cls(self, x, padding_mask):
        logits = list()
        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:,3]
        )
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return x[:,0]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward_finetuning_with_embeddings(self, x, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
        return x

    def forward_finetuning_with_embeddings_cls(self, x, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        logits = list()
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return torch.stack(logits, dim=0).mean(dim=0)

    def get_sequence_embedding(self, x):
        """Get only embeddings"""
        return self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            #log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

class MaskedLanguageModel(nn.Module):
    """Masked Language Head for Predictions"""

    def __init__(self, hparams, embedding, act:str = "tanh"):
        super(MaskedLanguageModel, self).__init__()
        self.hparams = hparams
        self.act = ACT2FN[act]
        self.dropout = nn.Dropout(p=self.hparams.emb_dropout)

        self.V = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.g = nn.Parameter(torch.tensor([hparams.hidden_size**0.5]))
        self.out = nn.Linear(
                self.hparams.hidden_size, 
                self.hparams.vocab_size, 
                bias=False
            )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            ignore_index = torch.LongTensor([0,4,5,6,7,8])
            log.info("(MLM Decoder) centering: true normalisation: %s" %hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(ignore_index = ignore_index, norm=hparams.norm_output_emb))

    def batched_index_select(self, x, dim, indx):
        """Gather the embeddings of tokens that we should make prediction on"""
        indx_ = indx.unsqueeze(2).expand(indx.size(0), indx.size(1), x.size(-1))
        return x.gather(dim, indx_)

    def forward(self, logits, batch): ##before 2.97
        indx = batch["target_pos"].long()
        logits = self.dropout(self.batched_index_select(logits, 1, indx))
        logits = self.dropout(l2_norm(self.act(self.V(logits))))
        return  self.g * self.out(logits)


class CLS_Decoder(nn.Module):
    """Classification for CLS Predictions"""
    def __init__(self, hparams):
        super(CLS_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = hparams.cls_num_targs
        p = hparams.dc_dropout

        self.in_layer = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act = ACT2FN["swish"]
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x = self.dropout(self.norm(self.act(self.in_layer(x))))
        return self.out_layer(x)

class CLS_Decoder_FT2(nn.Module):
    """Classification for CLS Predictions"""
    def __init__(self, hparams):
        super(CLS_Decoder_FT2, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = 2
        p = hparams.dc_dropout

        self.ff1 = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm_1 = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act_1 = Swish()#nn.Tanh()
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        #x = self.norm_1(self.act_1(self.ff1(self.dropout(x))))
        x = self.dropout(self.norm_1(self.act_1(self.ff1(x))))
        return self.out_layer(x)

class AttentionDecoderL(nn.Module):
    def __init__(self, hparams, num_outputs: int, num_heads: int = 10, init: bool = False) -> None:
        super().__init__()
        hidden_size = hparams.hidden_size
        self.num_targets= hparams.num_targets
        self.num_outputs = num_outputs
        context_size = hidden_size // 2
        self.num_heads = num_heads
        ## LAYERS
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.Linear(hidden_size, context_size)
        self.post = nn.Linear(hidden_size * num_heads, hidden_size)
        self.out = nn.Linear(hidden_size, num_outputs * self.num_targets)
        

        ## ACTIVATIONS
        self.act = Swish()
        self.tanh = nn.Tanh()
        self.sigsoftmax = SigSoftmax(dim = -1)
        ## MODULES
        self.dropout = nn.AlphaDropout(p=hparams.dc_dropout)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)
        self.rezero = ReZero(hidden_size=hidden_size)
        self.identity = nn.Identity()

        context = nn.Parameter(
                torch.randn((num_heads, context_size)))

        if init:
            nn.init.kaiming_uniform_(self.ff.weight) # as it is followed by relu-like activation
            nn.init.xavier_uniform_(self.out.weight, gain=1.)
            nn.init.xavier_uniform_(self.pool.weight, gain=1.)
            nn.init.xavier_uniform_(self.post.weight, gain=1.)
            nn.init.xavier_uniform_(context, gain=1.)
        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=context)

        self.attn = None

    def attention_pooling(self, x, mask):
        h = self.tanh(self.pool(x))  # Shape: [batch_size, sequence_len, hidden_size]
        context_expanded = self.context.unsqueeze(1).unsqueeze(1)
        h_expanded = h.unsqueeze(0)
        scores = torch.mul(h_expanded, context_expanded).sum(dim=-1, keepdim=False)

        # Apply mask and softmax to scores
        mask_expanded = mask.unsqueeze(0).bool()  # Expanding mask for all heads
        scores = self.sigsoftmax(scores, mask_expanded) 

        # Store attention for each head
        self.attn = scores  # Shape: [num_heads, batch_size, sequence_len]
        out = torch.mul(x.unsqueeze(0), scores.unsqueeze(-1)).sum(dim=2)

        out = out.permute(1, 0, 2)  # Change to [batch_size, num_heads, hidden_size]
        out = out.reshape(out.size(0), -1)  # Flatten last two dimensions

        return self.post(out)
    
    def forward(self, x, mask):
        logits = self.dropout(self.norm(self.act(self.ff(x))))
        logits = self.identity(self.attention_pooling(x = logits, mask = mask))
        o = self.out(logits) 
        return o.reshape(-1, self.num_targets, self.num_outputs)
    
    
class Flat_Decoder(nn.Module):
    """Classification for PSY Predictions"""
    def __init__(self, hparams, num_outputs: int):
        super(Flat_Decoder, self).__init__()
        hidden_size = hparams.hidden_size 
        num_targets = hparams.num_targets
        self.num_outputs = num_outputs
        self.num_targets = num_targets
        self.ff = nn.Linear(hidden_size, hparams.hidden_ff)
        self.ff2 =  nn.Linear(hparams.hidden_ff, hidden_size)
        self.dp_in = nn.Dropout(p = hparams.dc_dropout)
        self.dp_mid = nn.Dropout(p = hparams.dc_dropout)

        self.pool = nn.Linear(1,1)
        self.out = nn.Linear(hidden_size, num_outputs * num_targets, bias=False)

        self.act_in = Swish()
        self.act = Swish()
        self.act_out = Swish()
        self.norm = nn.LayerNorm(hparams.hidden_ff)

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(1)))

        self.register_parameter(name="context_1", param=nn.Parameter(
                torch.randn(1)))
        self.register_parameter(name="context_2", param=nn.Parameter(
                torch.randn(1)))

        self.register_parameter(name="context_3", param=nn.Parameter(
                torch.randn(1)))

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        y = x[:,0]
        x = self.act_in(y)
        x = self.dp_in(x)
        x = self.act(self.ff(x))
        x = self.norm(x)
        x = self.dp_mid(x)
        x = self.act_out(self.ff2(x))
        o = self.out(x) #* self.context
        
        return o.reshape(-1, self.num_targets, self.num_outputs)


class AttentionDecoder(nn.Module):
    def __init__(self, hparams, num_outputs:int) -> None:
        super().__init__()
        hidden_size = hparams.hidden_size
        context_size = hidden_size // 2
        ## LAYERS
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.Linear(hidden_size, context_size)
        #self.post = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_outputs)
        ## ACTIVATIONS
        self.act = Swish()
        self.tanh = nn.Tanh()
        self.sigsoftmax = SigSoftmax(dim = -1)
        ## MODULES
        self.dropout = nn.AlphaDropout(p=hparams.dc_dropout)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.identity = nn.Identity()

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(context_size)))

        self.attn = None
    def attention_pooling(self, x, mask):
        h = self.tanh(self.pool(x))
        scores = torch.mul(h, self.context) \
                      .sum(dim = -1, keepdim=False)
        scores = self.sigsoftmax(scores, mask.bool())
        self.attn = scores
        return  torch.mul(x, scores.unsqueeze(-1)).sum(dim = 1)

    def forward(self, x, mask):
        logits = self.dropout(self.norm(self.act(self.ff(x))))
        logits = self.identity(self.attention_pooling(x = logits, mask = mask))
        return self.out(logits)

