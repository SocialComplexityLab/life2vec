from performer_pytorch import SelfAttention
from performer_pytorch import performer_pytorch
from performer_pytorch.performer_pytorch import default, exists, rearrange, empty
import torch
import logging
log = logging.getLogger(__name__)


############################################################
# OVERWRITE THE PERFORMER IMPLEMENTATION (PERFORMER_PYTORCH)


def _orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'complete') 
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

#### OUR EDIT TO THE PACKAGE
# Overwrite the old Implementation of orthogonal matrix chunking (PyTorch issue)
performer_pytorch.orthogonal_matrix_chunk = _orthogonal_matrix_chunk

class CustomSelfAttention(SelfAttention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            try:
                self.local_attn.rel_pos = None
            except:
                log.warning("No Local Attention")
        def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, 
                    pos = None, 
                    pos_projection: bool = False, **kwargs):
            assert not exists(context), 'self attention should not receive context'
            b, n, _, h, gh = *x.shape, self.heads, self.global_heads

            cross_attend = False 

            context = default(context, x)
            context_mask = mask  # OUR EDIT: default(context_mask, mask) if not cross_attend else context_mask

            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

            ##### OUR EDITS TO THE PACKAGE:
            #if exists(pos):
            #    if pos_projection:
            #           pos = 
            #        q = self.sum(q,self.to_pos(pos))
            #    else:
            #        q = self.sum(q,pos)


            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
            (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
            attn_outs = []

            if not empty(q):
                if exists(context_mask):
                    global_mask = context_mask[:, None, :, None]
                    v.masked_fill_(~global_mask, 0.)

                ## OUR EDITS TO THE PACKAGE
                #if exists(pos_emb) and not cross_attend:
                #    q, k = apply_rotary_pos_emb(q, k, pos_emb)

                out = self.fast_attention(q, k, v)
                attn_outs.append(out)

            if not empty(lq):
                assert not cross_attend, 'local attention is not compatible with cross attention'
                out = self.local_attn(lq, lk, lv, input_mask = mask)
                attn_outs.append(out)

            out = torch.cat(attn_outs, dim = 1)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out =  self.to_out(out)
            return self.dropout(out)