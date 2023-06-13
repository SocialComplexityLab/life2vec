from turtle import pos
from torchmetrics import StatScores, Metric
from torchmetrics.utilities.enums import AverageMethod
from typing import Any, Dict, Optional, List
import torch
from torch import Tensor
import pandas as pd
import numpy as np
import os
from itertools import product
from torchmetrics.utilities.data import dim_zero_cat
import itertools

class CorrectedF1(StatScores):
    """Metric for Corrected F1-Score
       Based on 'Estimating classification accuracy in positive-unlabeled learning: characterization and correction strategies' """
    def __init__(
        self,
        alpha: float,
        beta: float,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro", ### equal weight for each class
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:

        assert beta >= 0. and beta <= 1, "Incorrect beta"
        assert alpha >= 0. and alpha <= 1, "Incorrect alpha"
        assert beta >= alpha, "Incorrect estimates of beta and alpha"
        self.beta = beta ## fraction of positives in Positive (P) set
        self.alpha = alpha ## fraction of positives in Unlabeled (U) set
        
        allowed_average = list(AverageMethod)
        if average not in allowed_average:
            raise ValueError(f"The àverage`has to be one of {allowed_average}, got {average}.")
        super().__init__(
            reduce= average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

    def _c_cr(self, c: Tensor):
        """Calculate corrected prior (Proportion of positive examples/targets)"""
        return c * self.beta + (1-c) * self.alpha
    
    def _gamma_cr(self, gamma: Tensor, nu: Tensor):
        """Calculate corrected True Positive rate"""
        return (self.beta - self.alpha)**-1 * \
             ((1-self.alpha) * gamma - (1-self.beta) * nu)

    def _nu_cr(self, gamma, nu):
        """Calculate corrected False Negative rate"""
        return ((self.beta - self.alpha)**-1) * \
            (self.beta * nu - self.alpha * gamma)

    def compute(self) -> Tensor:
        gamma_cr, _, c_cr, theta = self._calculate_parameters()
        return self._calculate_f1_corrected(gamma_cr=gamma_cr, 
                                            theta=theta, c_cr=c_cr)

    def _calculate_parameters(self):
        tp, fp, tn, fn = self._get_final_stats()
        sample_size = tp + fp + tn + fn
        gamma_hat = tp / (tp + fn) ## True Positive Rate
        nu_hat = fp/(tn+fp) ## False Positive Rate
        c = (tp + fn) / sample_size ## (pi) proportion of positive examples/targets
        theta = (tp + fp) / sample_size ## proportion of positive predictions (we do not need to correct it)

        gamma_cr = self._gamma_cr(gamma = gamma_hat, nu = nu_hat) ## corrected TPR
        nu_cr = self._nu_cr(gamma=gamma_hat, nu = nu_hat) ## corrected FNR
        c_cr = self._c_cr(c = c) ## corrected "prior"
        return gamma_cr, nu_cr, c_cr, theta ## corrected TPR, corrected FNR, corrected prior, % of positive predictions

    @staticmethod
    def _calculate_f1_corrected(gamma_cr: Tensor, 
                      theta: Tensor, 
                      c_cr:Tensor):
        """Return Corrected estimate of F1 Score"""
        return (2 * c_cr * gamma_cr) / (c_cr + theta)


class CorrectedBAcc(CorrectedF1):
    """Metric for Corrected Balanced Accuracy"""
    def compute(self) -> Tensor:
        gamma_cr, nu_cr, _, _ = self._calculate_parameters()
        return self._calculate_bacc_corrected(gamma_cr=gamma_cr, 
                                              nu_cr=nu_cr)

    @staticmethod
    def _calculate_bacc_corrected(gamma_cr, nu_cr):
        """Return Corrected estimate of the Balanced Accuracy"""
        return (1+gamma_cr - nu_cr) / 2


class CorrectedMCC(CorrectedF1):
    """Metric for Corrected Matthews Correlation Coefficient"""
    def compute(self) -> Tensor:
        gamma_cr, nu_cr, c_cr, theta = self._calculate_parameters()
        return self._calculate_mcc_corrected(gamma_cr=gamma_cr, 
                                              nu_cr=nu_cr, c_cr=c_cr,
                                              theta=theta)

    @staticmethod
    def _calculate_mcc_corrected(gamma_cr, nu_cr, c_cr, theta):
        """Return Corrected estimate of the 
        Matthews Correlation Coefficient"""
        return torch.sqrt( (c_cr * (1 - c_cr)) / (theta * (1 - theta)) ) * \
            (gamma_cr - nu_cr)



class AUL(Metric):
    higher_is_better: bool = True
    is_defferentiable: bool = False
    full_state_update: bool = False

    prb: List[Tensor]
    trg: List[Tensor]
    def __init__(self) -> None:
        super().__init__()

        self.add_state("prb", default=[], dist_reduce_fx="cat")
        self.add_state("trg", default=[], dist_reduce_fx="cat")


    def update(self, prb: Tensor, trg: Tensor):
        prb, trg = self._transform(prb, trg)
        self.prb.append(prb)
        self.trg.append(trg)

    def compute(self):
        prb = dim_zero_cat(self.prb)
        trg = dim_zero_cat(self.trg)
        prb_pos = prb[trg == 1]

        x = torch.cartesian_prod(prb_pos, prb)
        score = .0
        score += (x[:,0] > x[:, 1]).sum()
        score += (x[:,0] == x[:,1]).sum() * 0.5

        n_pos = float(sum(trg))
        n = float(trg.shape[0])
        return (score + 1e-7)/(n_pos*n) ## epsilon for stability


    @staticmethod
    def _transform(x: Tensor, y:Tensor):
        if x.ndim > 1:
            raise ValueError("Supply only probability of the Positive Class")
            #x = x.squeeze()

        if y.ndim > 1:
            y = y.squeeze()

        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("x and y should be 1d")
        
        if x.numel() != y.numel():
            raise ValueError("x and y do not have equal amount of elements")

        return x,y

class MetricTable:
    def __init__(self, path: str):
        """Initialise the Table"""
        # self.threshold =  np.array([0.1, 0.3 , 0.5, 0.7])
        self.threshold = np.array([0.5, 0.7])
        self.df = pd.DataFrame(
            0, index=self.init_indx(), columns=["TP", "FP", "TN", "FN"]
        )
        self.df = self.df.sort_index()
        self.path = path

    def init_indx(self):
        """Setup for a MultiIndex DataFrame"""
        groups = {
            "sex": np.array([1, 2]),
            "age": np.array([i for i in range(0, 101)]),
            "threshold": self.threshold,
        }

        indx = []
        for a in groups["age"]:
            for s in groups["sex"]:
                for t in groups["threshold"]:
                    indx.append((t, s, a))

        return pd.MultiIndex.from_tuples(indx, names=["threshold", "sex", "age"])

    def update(self, p_mortality, target, identifier):
        """Update the table with results"""
        sex, age = identifier
        if age > 100:
            age = 100

        for t in self.threshold:
            tp, tn, fp, fn = self.calculate(p_mortality, target, t)
            self.df.loc[(t, sex, age)]["TP"] += tp
            self.df.loc[(t, sex, age)]["TN"] += tn
            self.df.loc[(t, sex, age)]["FP"] += fp
            self.df.loc[(t, sex, age)]["FN"] += fn

    def calculate(self, p, target, threshold):
        """Return the metric"""
        prediction = int(p > threshold)
        ## reutrn TP, TN, FP, FN
        if prediction == target and target == 1:
            return 1, 0, 0, 0
        elif prediction == target and target == 0:
            return 0, 1, 0, 0
        elif prediction != target and target == 1:
            return 0, 0, 0, 1
        elif prediction != target and target == 0:
            return 0, 0, 1, 0
        else:
            raise Exception("Wrong values")

    def to_csv(self, name: str = "0"):
        try:
            os.makedirs(self.path)
        except Exception as e:
            pass
        self.df.to_csv(self.path + "MetricTable1_%s.csv" % name)

    def empty(self):
        """Set all the values to 0"""
        self.df[:] = 0
