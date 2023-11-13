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

class CLS_Decoder_FT(nn.Module):
    """Classification for CLS Predictions"""
    def __init__(self, hparams):
        super(CLS_Decoder_FT, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = 2
        p = hparams.dc_dropout

        self.ff1 = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm_1 = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.ff2 = nn.Linear(hidden_size, hidden_size//2)
        self.norm_2 = ScaleNorm(hidden_size=hidden_size//2, eps=hparams.epsilon)

        self.act_1 = Swish()
        self.act_2 = Swish()
        self.out_layer = nn.Linear(hidden_size //2, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x = self.dropout(self.norm_1(self.act_1(self.ff1(x))))
        x = self.norm_2(self.act_2(self.ff2(x)))
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

class PSY_Decoder(nn.Module):
    """Classification for PSY Predictions"""
    def __init__(self, hparams, num_outputs: int):
        super(PSY_Decoder, self).__init__()
        hidden_size = hparams.hidden_size 
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.Linear(1,1)
        self.out = nn.Linear(hidden_size, num_outputs)
        self.act = Swish()

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(1)))

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x = self.act(self.ff(x[:,0]))
        return self.out(x)


class PSY_Wide_Decoder(nn.Module):
    """Classification for PSY Predictions"""
    def __init__(self, hparams, num_outputs: int):
        super(PSY_Wide_Decoder, self).__init__()
        hidden_size = hparams.hidden_size 
        self.norm = nn.Identity()#ScaleNorm(hidden_size=hidden_size)
        self.ff = nn.Linear(hidden_size, hidden_size * 5)
        self.dropout_in = nn.Dropout(p = hparams.dc_dropout)

        self.dropout_ff = nn.Dropout(p = hparams.dc_dropout)
        self.pool = nn.Linear(1,1)
        self.out = nn.Linear(hidden_size * 5, num_outputs)
        self.act = Swish()
        

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(1)))

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x =  self.norm(self.dropout_in(x[:,0]))
        x = self.act(self.ff(x))
        x = self.dropout_ff(x)
        return self.out(x)


class Wide_Decoder(nn.Module):
    """Classification for PSY Predictions"""
    def __init__(self, hparams, num_outputs: int):
        super(Wide_Decoder, self).__init__()
        hidden_size = hparams.hidden_size 
        self.norm = ScaleNorm(hidden_size)
        self.ff = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout_in = nn.Dropout(p = hparams.dc_dropout)

        self.dropout_ff = nn.Dropout(p = hparams.dc_dropout)
        self.pool = nn.Linear(1,1)
        self.out = nn.Linear(hidden_size , num_outputs)
        self.act = nn.Swish()
        

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(1)))

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x =  self.dropout_in(self.norm(x[:,0]))
        x = self.act(self.ff(x))
        
        return self.out(x)

class Deep_Decoder(nn.Module):
    """Classification for PSY Predictions"""
    def __init__(self, hparams, num_outputs: int):
        super(Deep_Decoder, self).__init__()
        hidden_size = hparams.hidden_size 
        self.ff = nn.Linear(hidden_size, hparams.hidden_ff)
        self.ff2 =  nn.Linear(hparams.hidden_ff, hidden_size)
        self.dp_in = nn.Dropout(p = hparams.dc_dropout)
        self.dp_mid = nn.Dropout(p = hparams.dc_dropout)

        self.pool = nn.Linear(1,1)
        #self.out = nn.utils.weight_norm(nn.Linear(hidden_size, num_outputs, bias=False))
        self.out = nn.Linear(hidden_size, num_outputs, bias=False)

        self.out_1 = nn.Linear(hidden_size, num_outputs, bias=False)
        self.out_2 = nn.Linear(hidden_size, num_outputs, bias=False)
        self.out_3 = nn.Linear(hidden_size, num_outputs, bias=False)
        self.out_4 = nn.Linear(hidden_size, num_outputs, bias=False)
        self.out_5 = nn.Linear(hidden_size, num_outputs, bias=False)

        
        #parametrize.register_parametrization(self.out, "weight", Center(ignore_index = torch.LongTensor([0]), use_ignore_index=False,
        #                                 norm=True))
        #parametrize.register_parametrization(self.out_1, "weight", Center(ignore_index = torch.LongTensor([0]), use_ignore_index=False,
        #                                 norm=True))
        #parametrize.register_parametrization(self.out_2, "weight", Center(ignore_index = torch.LongTensor([0]), use_ignore_index=False,
        #                                 norm=True))
        #parametrize.register_parametrization(self.out_3, "weight", Center(ignore_index = torch.LongTensor([0]), use_ignore_index=False,
        #                                 norm=True))

        self.act_in = Swish()
        self.act = Swish()
        self.act_out = Swish()
        self.norm = nn.LayerNorm(hparams.hidden_ff)
        #self.out = nn.utils.weight_norm(nn.Linear(hidden_size, 20, bias=False))

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
        #x = self.norm_in(self.act_in(x[:,0]))
        #bs = x.shape[0]
        x = self.act_in(x[:,0])
        x = self.dp_in(x)
        x = self.act(self.ff(x))
        x = self.norm(x)
        x = self.dp_mid(x)
        #x = l2_norm(self.act_out(self.ff2(x)))
        x = self.act_out(self.ff2(x))
        o0 = self.out(x) #* self.context
        o1 = self.out_1(x) #* self.context_1
        o2 = self.out_2(x)# * self.context_2
        o3 = self.out_3(x)# * self.context_3
        o4 = self.out_4(x)
        o5 = self.out_5(x)

        x = torch.stack([o0, o1, o2, o3, o4, o5]).permute(1,0,2)
        return x
        #x = l2_norm(x.view(bs, 4, -1))
        return self.out(x).view(bs, 4,-1) # * self.context

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
        #self.rezero = ReZero(hidden_size=hidden_size)

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
        #x = self.norm_in(self.act_in(x[:,0]))
        #bs = x.shape[0]
        y = x[:,0]
        x = self.act_in(y)
        x = self.dp_in(x)
        x = self.act(self.ff(x))
        x = self.norm(x)
        x = self.dp_mid(x)
        #x = l2_norm(self.act_out(self.ff2(x)))
        x = self.act_out(self.ff2(x))
        #x = self.rezero(x,y)
        o = self.out(x) #* self.context
        
        return o.reshape(-1, self.num_targets, self.num_outputs)



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



class AttentionDecoderP(nn.Module):
    def __init__(self, hparams, num_outputs:int) -> None:
        super().__init__()
        hidden_size = hparams.hidden_size
        context_size = hidden_size // 2
        ## LAYERS
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.Linear(hidden_size, context_size)
        #self.post = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_outputs )

        self.out_1 = nn.Linear(hidden_size, num_outputs )
        self.out_2 = nn.Linear(hidden_size, num_outputs)
        self.out_3 = nn.Linear(hidden_size, num_outputs)
        self.out_4 = nn.Linear(hidden_size, num_outputs)
        self.out_5 = nn.Linear(hidden_size, num_outputs)
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


        o0 = self.out(logits) #* self.context
        o1 = self.out_1(logits) #* self.context_1
        o2 = self.out_2(logits)# * self.context_2
        o3 = self.out_3(logits)# * self.context_3
        o4 = self.out_4(logits)
        o5 = self.out_5(logits)

        x = torch.stack([o0, o1, o2, o3, o4, o5]).permute(1,0,2)
        return x


        

class MosDecoder(nn.Module):
    def __init__(self, hparams, num_outputs:int, k: int) -> None:
        super().__init__()
        d = 0.01
        self.hidden_size = hparams.hidden_size
        self.k = k
        self.num_outputs = num_outputs
        self.norm = ScaleNorm(hidden_size=self.hidden_size)
        self.ff = nn.Linear(self.hidden_size, self.hidden_size * self.k) # P ¨
        self.M = nn.Linear(self.hidden_size, self.k)

        self.out = nn.Linear(self.hidden_size, self.num_outputs)

        self.tanh = nn.Tanh()
        self.softmax = SigSoftmax(dim=-1)


        self.drop_ff = nn.Dropout(p=hparams.dc_dropout)
        self.drop_m = nn.Dropout(p= hparams.dc_dropout)

        torch.nn.init.xavier_normal_(self.ff.weight, gain = 5/3)
        torch.nn.init.uniform_(self.M.weight, a = -d , b = -d)
        torch.nn.init.uniform_(self.out.weight, a = -d , b = -d)

        ## PLACEHOLDERS
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(1)))
        self.pool = nn.Linear(1,1) # placeholder

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        logit = self.norm(x[:,0])
        h = self.tanh(self.ff(logit)) # size: (BS, HS * k)
        h = h.view(-1, self.hidden_size)  # size: (BS * k, HS)
        h = self.drop_ff(h)

        m = self.softmax(self.M(logit))# size: (BS, k)
        #m = self.drop_m(m)

        out = self.softmax(self.out(h))#self.softmax(self.out(h)) # size (BS * k, C)

        out = out.view(-1, self.k, self.num_outputs) # size: (bs, k, c)
        #print(out[0])
        #print(m[0])
        
        #print(out.sum(-1))
        out = torch.einsum("bkc, bk-> bc", out, m)
        #print(out.sum(-1))
        return out # size: (bs, c)

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




class CLS_Decoder_FT3(nn.Module):
    """Classification for CLS Predictions"""
    def __init__(self, hparams):
        super(CLS_Decoder_FT3, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = 2
        p = hparams.dc_dropout

        self.ff1 = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm_1 = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act_1 = Swish()
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        x = self.act_1(self.ff1(x))
        return self.out_layer(x)


class Causal_Norm_Classifier(nn.Module):
    
    def __init__(self, hparams, num_outputs: int, use_effect=True, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125, *args):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_outputs, hparams.hidden_size), requires_grad=True)
        self.register_parameter("weight", param= nn.Parameter(torch.Tensor(num_outputs, hparams.hidden_size)))
        self.feature_mean = torch.zeros(hparams.hidden_size, requires_grad=False)
        self.mu = 0.9
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = hparams.hidden_size // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        self.act = Swish()
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def update_mean(self, x):
        self.feature_mean = self.feature_mean * self.mu + x.detach().sum(0).view(-1)

    def forward(self, x, **kwargs):
        if self.training:
            self.update_mean(x)
        x = x[:,0].view(-1, self.num_head, self.head_dim)
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            normed_c = self.multi_head_call(self.l2_norm, self.feature_mean)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y = sum(output)
            
        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x
