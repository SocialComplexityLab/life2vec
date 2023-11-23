from signal import Sigmasks
from unicodedata import unidata_version
from focal_loss.focal_loss import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import math
import logging

from src.transformer.attention import MultiHeadAttention
import torch
from torch import nn
from typing import Optional
import numpy as np


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss

    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')



def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights  = None
                         ) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.

    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.

    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    Returns
    -------
    loss: torch.Tensor

    """
    eps = 1e-15

    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss

def masked_sort(x, mask, dim=-1, descending = True):
    mask = mask.type(x.dtype)
    masked = torch.mul(x,mask)
    neg_inf = torch.zeros_like(x).to(masked.device).masked_fill(mask == 0, -math.inf)
    return torch.sort((masked + neg_inf), dim=dim, descending=descending)[0]

def masked_max(x, mask, dim= -1):
    mask = mask.type(x.dtype)
    masked = torch.mul(x,mask)
    neg_inf = torch.zeros_like(x).to(masked.device).masked_fill(mask == 0, -math.inf)
    return torch.max((masked + neg_inf), dim=1)[0]

def hard_softmax(x, mask, dim=-1):
    n_logits = x.shape[dim]
    batch_size = x.shape[0]
    x_centered = x - masked_max(x, mask).view(batch_size, -1).expand_as(x)
    x_sorted = masked_sort(x_centered, mask)
    _range = torch.arange(start = 1, end = n_logits + 1, step = 1, device = x.device, dtype = x.dtype).view(1,-1)
    _range = _range.expand_as(x)
    
    bound = 1 + _range * x_sorted
    cumsum = torch.cumsum(x_sorted, dim=-1)
    
    is_gt = torch.gt(bound, cumsum)
    x_sparse = is_gt * x_sorted
    k = torch.max(is_gt * _range, dim = -1, keepdim=True)[0]
    taus = (torch.nansum(x_sparse, dim=-1, keepdim=True) - 1) / k

    x_adj = x_centered - taus 
    x_adj = torch.mul(x_adj, mask)
    

    return torch.max(torch.zeros_like(x), x_adj)


class Swish(nn.Module):
    def forward(self, input: Tensor):
        return swish(input)

def l2_norm(x):
    return F.normalize(x, dim=-1, p=2)

log = logging.getLogger(__name__)

class ReProject(nn.Module):
    def __init__(self, hidden_size, activation) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.eye(hidden_size)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(hidden_size)))
        self.act = activation
    def forward(self, X):
        return self.act(F.linear(input=l2_norm(X), weight=self.w, bias=self.bias))

class Center(nn.Module):
    def __init__(self,  ignore_index: torch.LongTensor, norm: bool = False, use_ignore_index: bool = True) -> None:
        super().__init__()
        self.register_buffer("norm", torch.BoolTensor([norm]))
        self.register_buffer("use_ignore_index", torch.BoolTensor([use_ignore_index]))

        self.register_buffer("ignore_index", ignore_index)

    def forward(self, X):
        if self.use_ignore_index:
            mask = self.mask(X)
            X = X - X[mask].mean(0) ## we do not want tokens as PAD and PLACEHOLDERS to contribute to the mean 
            X[self.ignore_index] *= 0 ### we do not want to do anything with the indexes we ignore, like PAD and PLACEHOLDERS
        else:
            X = X - X.mean(0)
        if self.norm:
            return l2_norm(X)
        return X
    
    def mask(self,X):
        mask = torch.ones(X.shape[0])
        mask[self.ignore_index] = 0
        return mask.bool()

class Norm(nn.Module):
    def forward(self, X):
        return l2_norm(X)


class FixNormaliseWeights(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.norm = FixNorm(hidden_size)
    def forward(self, X):
        return self.norm(X)


#######################
# Activation Functions
#######################

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


def cosine_annealing(current_step):
    """Cosine Annealing for the Learning Rate"""
    # max_training_steps = 1000
    # progress = min(current_step, max_training_steps * .9)/max_training_steps #+ 1e-5
    progress = min(current_step * 0.033, 0.95)
    return math.cos(0.5 * math.pi * progress)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
}

###################################
## Normalisation and Residuals
###################################

class Gate(torch.nn.Module):
    """
    Gating mechanism for the residual layers: "Highway Transformer" implementation
    """

    def __init__(self, hidden_size, bias: int = -2):
        """"""
        super(Gate, self).__init__()
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ug = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_bias(bias)
        # self.Wz.bias.data.fill_(bias)


    def init_bias(self, bias):
        """"""
        with torch.no_grad():
            self.Wz.bias.fill_(
                bias
            )  ## STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING according to the paper

    def forward(self, x, y):
        """"""
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x))

        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class ReZero(torch.nn.Module):
    def __init__(self, hidden_size, simple: bool = True, fill:float =.0):
        """"""
        super(ReZero, self).__init__()
        if simple: ## aka original
            self.weights = torch.nn.Parameter(torch.add(torch.zeros(1), fill))
        else:
            self.weights = torch.nn.Parameter(torch.add(torch.zeros(1), fill))

    def forward(self, x, y):
        return x + y * self.weights


class ScaleNorm(torch.nn.Module):
    """L2-norm (Alternative to LayerNorm)"""

    def __init__(self, hidden_size, eps=1e-6):
        """"""
        super(ScaleNorm, self).__init__()
        self.g = torch.nn.Parameter(torch.sqrt(torch.Tensor([hidden_size])))
        self.eps = eps


    def forward(self, x):
        """"""
        norm = self.g / torch.linalg.norm(x, dim=-1, ord=2, keepdim=True).clamp(
            min=self.eps
        )
        return x * norm


class FixNorm(ScaleNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.g.weight = torch.Tensor([1.])
        self.g.requires_grad = False



class SigSoftmax(nn.Module):
    """Implementation of SigSoftmax (prevents oversaturation)"""
    def __init__(self, dim: int = -1, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, -np.inf)
        return self.softmax(x + torch.log(torch.sigmoid(x) + self.epsilon))


###############################
## Components and Modules
###############################


class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer normalisation
    """

    def __init__(
        self, hparams
    ):
        """"""
        super(SublayerConnection, self).__init__()
        assert hparams.norm_type in ["pre_norm", "rezero"]

        self.norm_type = hparams.norm_type
        hidden_size = hparams.hidden_size

        if self.norm_type == "rezero":
            self.norm = ReZero(hidden_size)
        elif self.norm_type == "pre_norm":
            self.norm = ScaleNorm(hidden_size)
            self.gate = Gate(hidden_size)

    def forward(self, x, sublayer, **kwargs):
        """
        Apply a residual connection to a sublayer"
        """
        if self.norm_type == "rezero":
            """
            ReZero
            """
            return self.norm(x, sublayer(x, **kwargs))
        elif self.norm_type == "pre_norm":
            """
            PRE NORM (ScaleNorm + Gate)
            """
            return self.gate(x, sublayer(self.norm(x), **kwargs))


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """

    def __init__(self, hparams):
        """"""
        super(PositionWiseFeedForward, self).__init__()
        self.hidden2ff = nn.Linear(hparams.hidden_size, hparams.hidden_ff)
        self.dropout = nn.Dropout(hparams.fw_dropout)
        self.act = ACT2FN[hparams.hidden_act]
        self.ff2hidden = nn.Linear(hparams.hidden_ff, hparams.hidden_size)

    def forward(self, x):
        """"""
        x = self.act(self.dropout(self.hidden2ff(x)))
        return self.ff2hidden(x)

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
        self.attention_sublayer = SublayerConnection(hparams)

        self.position_wise = PositionWiseFeedForward(hparams)
        self.position_sublayer = SublayerConnection(hparams)

        log.info("EncoderLayer setup is finised:  %.3f s" % (time.time() - start))

    def redraw_projection_matrix(self):
        """Redraw projection matrices during the training"""
        try:
            try:
                self.attention.attention.fast_attention.redraw_projection_matrix("cuda")
            except:
                self.attention.attention.fast_attention.redraw_projection_matrix("cpu")
        except:
            log.warning("Cannot redraw random projections. Wrong attention type")

    def forward(self, x, mask=None):
        """Forward Pass"""
        x = self.attention_sublayer(x, sublayer=self.attention, mask=mask)
        x = self.position_sublayer(x, sublayer=self.position_wise)

        return x

###############
## Loss Fn
###############
class CDW_CELoss(nn.Module):
    def __init__(self, num_classes, alpha= 2., delta = 3.,
                 reduction: str = "mean",
                 transform: str = "huber",
                 eps: float = 1e-5):
        super(CDW_CELoss, self).__init__()
        assert alpha > 0, "Alpha should be larger than 0"
        self.reduction = reduction
        self.transform = transform
        self.alpha = alpha
        self.eps = eps
        self.num_classes = num_classes
        self.register_buffer(name="w", tensor=torch.tensor([float(i) for i in range(self.num_classes)]))
        self.softmax = SigSoftmax()

        self.delta = delta # for huber transform only

    def huber_transform(self, x):
        return torch.where(
            x< self.delta,
            0.5 * torch.pow(x, 2),
            self.delta * (x - 0.5 * self.delta)
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        w = torch.abs(self.w - target.view(-1,1))


        if self.transform == "huber":
            w = self.huber_transform(w)
        elif self.transform == "log":
            w = torch.log1p(w)
            w = torch.pow(w, self.alpha)
        elif self.transform == "power":
            w = torch.pow(w, self.alpha)
        else:
             raise NotImplementedError("%s transform is not implemented" %self.transform)

   
        loss = - torch.mul(torch.log(1 - logits + self.eps), w).sum(-1)
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError("%s reduction is not implemented" %self.reduction)
        

class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        self.register_parameter("noise_params", nn.Parameter(torch.full((n_losses,), 1./float(n_losses))))

    def forward(self, losses):
        output = [torch.exp(-self.noise_params[i]) * loss + self.noise_params[i] for i, loss in enumerate(losses)]
        return sum(output)
        
        
        
## BASED on https://github.com/rickgroen/cov-weighting
class CoVWeightingLoss(nn.Module):
    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self, num_losses: int, decay: bool = True, decay_param: float = 0.9):
        super(CoVWeightingLoss, self).__init__()
        # How to compute the mean statistics: Full mean or decaying mean.
        self.decay_flag = decay

        self.num_losses = num_losses
        self.register_buffer("decay_param", torch.FloatTensor([decay_param]))
        self.register_buffer("alphas", torch.ones((num_losses, )).type(torch.FloatTensor))
        self.register_buffer("current_iter", torch.Tensor([-1]))


        # Initialize all running statistics at 0.
        self.register_buffer("running_mean_L", torch.zeros((num_losses,)).type(torch.FloatTensor))
        self.register_buffer("running_mean_l", torch.zeros((num_losses,)).type(torch.FloatTensor))
        self.register_buffer("running_S_l", torch.zeros((num_losses,)).type(torch.FloatTensor))
        self.register_buffer("zero", torch.zeros((1)).type(torch.FloatTensor))
        self.running_std_l = None
        self.train_flag = False

    def set_train(self, x: bool):
        self.train_flag = x

    @property
    def iteration(self):
        return self.current_iter.type(torch.FloatTensor)


    def forward(self, unweighted_losses):
        device = unweighted_losses[0].device
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train_flag:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0
        # If we are in the first iteration set alphas to all 1/32

        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = self.zero.to(device)
        elif self.current_iter > 0 and self.decay_flag:
            mean_param = self.decay_param.to(device)
        else:
            mean_param = (1. - 1 / (self.iteration + 1))

        self.running_mean_l = self.running_mean_l.to(device)
        self.running_mean_L = self.running_mean_L.to(device)
        self.running_S_l = self.running_S_l.to(device)
        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.iteration.to(device) + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach().to(device)
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L



        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss



class AsymmetricCrossEntropyLoss(nn.Module):
    """CrossEntropy Loss for Positive-Unlabeled Learning
    Args:
    """
    def __init__(self, pos_weight: float = 0.5, penalty: float = 0., sigmoid: bool = False):
        super().__init__()
        #self.softmax = nn.Softmax(dim=1)
        #self.loss_positive = nn.CrossEntropyLoss(ignore_index=0 , reduction="sum")
        #self.loss_negative = nn.CrossEntropyLoss(ignore_index=1, reduction="sum")
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.ls = nn.LogSigmoid()
        else:
            self.ls= nn.LogSoftmax(dim = 1)
        self.loss = nn.NLLLoss()
        self.register_buffer("penalty", torch.tensor([penalty, 0.]))
        self.register_buffer("weight", torch.tensor([1-pos_weight, pos_weight]))

    def __calculate_loss__(self, loss_u, loss_p, n_u, n_p):
        loss_u = loss_u * self.weight[0]
        loss_p = loss_p * self.weight[1]
        if n_p == 0: 
            return loss_u
        elif n_u == 0:
            return loss_p
        else:
            return loss_p + loss_u
    
    def set_penalty(self, penalty):
        self.penalty[0] = penalty
    
    def adjust_penalty(self):
        self.penalty[0] = torch.addself.penalty[0] * 0.9 + 0.1

    @property
    def biased_penalty(self):
        """Penalty for ACE loss without SCAR assumption"""
        penalty = self.penalty 
        penalty[0] = penalty[0] / (1.0 - penalty[0])
        return penalty

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
        logits: raw logits (batch_size, num_classes)
        targets: one hot encoded (batch_size, num_classes)
        """
        n_p = target[:,1].sum()
        n_u = target.shape[0] - n_p
        is_positive, is_negative = (target[:,1] == 1).type(logits.dtype), (target[:,0] == 1).type(logits.dtype)

        if self.sigmoid:
            ## logits +=1
            return F.binary_cross_entropy_with_logits(logits.squeeze(), target=target[:,1].squeeze().float(), reduction="mean")
        scores = self.ls(logits)
        loss_p =  F.nll_loss(scores, target = target[:,1], ignore_index = 0, reduction="mean")
            #scores = logits + self.penalty.expand(scores.shape[0], 2)
        scores = self.ls(logits)
        loss_u = F.nll_loss(scores, target = target[:,1], ignore_index=1, reduction="mean")
        return self.__calculate_loss__(loss_u=loss_u,
                                       loss_p=loss_p,
                                       n_p=n_p, n_u = n_u)

        #else: ## without SCAR assumption
        #    scores = self.ls(logits + self.biased_penalty)

class RobustCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss class for noisy labels
    Args:
        T (Tensor): Row-Stochaistic transition matrix for the noise, shape (CxC)
        roobust_method (str): Specifies the method fo the robustness (either "forward" or "backward")
    """
    def __init__(self, T: Tensor = None,
                 robust_method: str = "backward") -> None:
        super(RobustCrossEntropyLoss, self).__init__()
        self.robust_method = robust_method
        assert self.robust_method in ["forward", "backward"]
        if self.robust_method == "backward":
            self.register_buffer("T", torch.linalg.inv(T))
        else:
            self.register_buffer("T", T)

    def forward(self, pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
        target = target.type(self.T.dtype)
        if self.robust_method == "backward":
            target = torch.inner(target, self.T)
            pred = torch.nn.functional.log_softmax(pred, dim = -1)
        else:
            pred = torch.clamp(pred.softmax(-1), min = eps, max = 1-eps)
            pred = torch.inner(pred, self.T)
            pred = torch.log(pred)

        return - torch.mean(torch.sum(target * pred, axis = -1))


class CumulativeLinkLoss(nn.Module):
    """
    BASED on SPACECUTTER Implementation
    Module form of cumulative_link_loss() loss function

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        y_pred = F.softmax(y_pred, dim=1)

        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)


def probability_recalibration(probs, pos_true_fraction, pos_train_fraction):
    pos_scaler = pos_true_fraction/pos_train_fraction
    neg_scaler = (1-pos_true_fraction)/(1-pos_train_fraction)
    output= torch.zeros_like(probs)

    output[:,1] = (probs[:,1] * pos_scaler) / ( (probs[:,1]) * pos_scaler) + (probs[:,0] * neg_scaler)
    output[:,0] = 1-output[:,1]
    return output


