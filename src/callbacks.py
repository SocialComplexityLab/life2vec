from asyncio.log import logger
from copy import copy
from pstats import Stats
from turtle import width
from typing import Any, Dict
import seaborn as sns
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import RandomSampler, WeightedRandomSampler
from coral_pytorch.dataset import corn_label_from_logits

import os
import hashlib
import json
import logging
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

from pathlib import Path
import numpy as np
from scipy.stats import binom
import scipy
import math
from scipy.optimize import fsolve
import random

from src.transformer.transformer_utils import SigSoftmax



class RiskControl:
    """Uncertainty estimation to identify the low-confidence prediction region
       Based on the implementation from: https://github.com/luoyan407/predict_trustworthiness"""
    def __init__(self, rstar: float, delta: float = 0.001, threshold: float = 0.5, split: bool = False) -> None:
        self.delta = delta
        self.rstar = rstar
        self.threshold = threshold
        self.split = split
    def run(self, preds, targs):
        kappa = np.zeros_like(preds)
        for i in range(kappa.shape[0]):
            if preds[i] > self.threshold:
                kappa[i] = preds[i]
            else:
                kappa[i] = 1 - preds[i]


        residuals = ((preds > self.threshold).astype(float) != targs).astype(float)[:, np.newaxis]
        return self.bound(self.rstar, self.delta,kappa,residuals, split=self.split)

    def calculate_bound(self,delta,m,erm):
        #This function is a solver for the inverse of binomial CDF based on binary search.
        precision = 1e-8
        def func(b):
            return (-1*delta) + scipy.stats.binom.cdf(int(m*erm),m,b)
        a=erm #start binary search from the empirical risk
        c=1   # the upper bound is 1
        b = (a+c)/2 #mid point
        funcval  =func(b)
        while abs(funcval)>precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval>0:
                a=b
            else:
                c=b
            b = (a + c) / 2
            funcval= func(b)
        return b

    def bound(self,rstar,delta,kappa,residuals,split=True, random_seed: int = 0):
        random.seed(random_seed)
        # A function to calculate the risk bound proposed in the paper, the algorithm is based on algorithm 1 from the paper.
        #Input: rstar - the requested risk bound
        #       delta - the desired delta
        #       kappa - rating function over the points (higher values is more confident prediction)
        #       residuals - a vector of the residuals of the samples 0 is correct prediction and 1 corresponding to an error
        #       split - is a boolean controls whether to split train and test
        #Output - [theta, bound] (also prints latex text for the tables in the paper)
        # when spliting to train and test this represents the fraction of the validation size
        valsize = 0.7
        probs = kappa
        FY = residuals
        if split:
            idx = list(range(len(FY)))
            random.shuffle(idx)
            slice = round(len(FY)*(1-valsize))
            FY_val = FY[idx[slice:]]
            probs_val = probs[idx[slice:]]
            FY = FY[idx[:slice]]
            probs = probs[idx[:slice]]
        m = len(FY)
        probs_idx_sorted = np.argsort(probs)
        a=0
        b = m-1
        deltahat = delta/math.ceil(math.log2(m))
        for q in range(math.ceil(math.log2(m))+1):
            # the for runs log(m)+1 iterations but actually the bound calculated on only log(m) different candidate thetas
            mid = math.ceil((a+b)/2)
            mi = len(FY[probs_idx_sorted[mid:]])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[probs_idx_sorted[mid:]])/mi
            if split:
                eps = 1e-10
                testrisk = sum(FY_val[probs_val>=theta])/(len(FY_val[probs_val>=theta]) + eps)
                testcov = len(FY_val[probs_val>=theta])/(len(FY_val) + eps)
            bound = self.calculate_bound(deltahat,mi,risk)
            coverage = mi/m
            if bound>rstar:
                a=mid
            else:
                b=mid
        if split:
            return {"rstar": rstar,
                    "risk": risk,
                    "coverage": coverage,
                    "testrisk": testrisk,
                    "testcov": testcov, 
                    "bound": bound,
                    "theta": theta}
        return {"rstar": rstar,
                    "risk": risk,
                    "coverage": coverage,
                    "testrisk": 0,
                    "testcov": 0, 
                    "bound": bound, 
                    "theta": theta}


HOME_PATH = str(Path.home())

log = logging.getLogger(__name__)

def get_hash(obj):
    bytes_repr = json.dumps(
        obj, 
        separators=(",", ":"),
        sort_keys=True,
        ).encode()

    return int(hashlib.shake_128(bytes_repr).hexdigest(4), base=16)

#class AsymLossTunning(pl.Callback):
#    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#        metrics = copy.deepcopy(trainer.callback_metrics)
#        acc = metrics["train/accuracy_epoch"]
#        fraction_of_positives = metrics["train/accuracy_epoch"]
#        if acc > (1- (0.5 - fraction_of_positives)):
#            pass
#        return super().on_train_epoch_end(trainer, pl_module)

class SaveWeights(pl.Callback):
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _id = "%s" %str(pl_module.hparams.version).replace(".","_")
        path = HOME_PATH + "/odrev/projekter/PY000017_D/weights/%s/%s/%s/" %(pl_module.hparams.implementation, 
                                                                             pl_module.hparams.training_task,
                                                                             pl_module.hparams.experiment_name)
        try:
            os.makedirs(path)
        except:
            pass
        torch.save(pl_module.transformer.state_dict(), path + _id + ".pth")
        log.info("Transformer weights saved:\n\t %s" %path)
        return super().on_fit_end(trainer, pl_module)
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _id = "%s" %str(pl_module.hparams.version).replace(".","_")
        path = HOME_PATH + "/odrev/projekter/PY000017_D/weights/%s/%s/%s/" %(pl_module.hparams.implementation, 
                                                                             pl_module.hparams.training_task,
                                                                             pl_module.hparams.experiment_name)
        try:
            os.makedirs(path)
        except:
            pass
        torch.save(pl_module.transformer.state_dict(), path + _id + ".pth")
        log.info("Transformer weights saved:\n\t %s" %path)
        return super().on_test_start(trainer, pl_module)

class RedrawRandomProjections(pl.Callback):
    """Performer Specific Callback to redraw Random Orthogonal projections."""
    def __init__(self) -> None:
        super().__init__()
        self.last_update_step = 0
    def on_train_epoch_end(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.hparams.attention_type == "performer":
            if (trainer.current_epoch + 1) % 3 == 0 :
                log.info("Redraw Projection Matrices")
                pl_module.transformer.redraw_projection_matrix(-1)


class CollectOutputs(pl.Callback):
    def on_validation_epoch_end(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            context = pl_module.decoder.context
            trainer.logger.experiment.add_histogram(tag="context", values=context, global_step=trainer.current_epoch)
        except: 
            context = pl_module.decoder.context.weight
            trainer.logger.experiment.add_histogram(tag="context", values=context, global_step=trainer.current_epoch)

        trainer.logger.experiment.add_histogram(tag="decoder_out", values=pl_module.decoder.out.weight, global_step=trainer.current_epoch)
        trainer.logger.experiment.add_histogram(tag="decoder_in", values=pl_module.decoder.ff.weight, global_step=trainer.current_epoch)
        trainer.logger.experiment.add_histogram(tag="decoder_query", values=pl_module.decoder.pool.weight, global_step=trainer.current_epoch)


    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            print(trainer.datamodule.test_dataloader().sampler.indices)
            path = HOME_PATH + "/odrev/projekter/PY000017_D/predictions/%s/%s/%s/%s/" %(pl_module.hparams.implementation, 
                                                                             pl_module.hparams.training_task,
                                                                             pl_module.hparams.experiment_name,
                                                                             pl_module.hparams.experiment_version)
            try:
                os.makedirs(path)
            except:
                pass

            with open(path + "test_data_indices.npy", "wb") as f:
                res = trainer.datamodule.test_dataloader().sampler.indices.numpy()
                np.save(f, res)
        except:
            print("Collect outputs disabled")

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            path = HOME_PATH + "/odrev/projekter/PY000017_D/predictions/%s/%s/%s/%s/" %(pl_module.hparams.implementation, 
                                                                             pl_module.hparams.training_task,
                                                                             pl_module.hparams.experiment_name,
                                                                             pl_module.hparams.experiment_version)
            try:
                os.makedirs(path)
            except:
                pass

            with open(path + "trg.npy", "wb") as f:
                trg = pl_module.test_trg.compute().cpu().numpy()
                np.save(f, trg)
            with open(path + "prb.npy", "wb") as f:
                prb = pl_module.test_prb.compute().cpu().numpy()
                np.save(f, prb)
            with open(path + "id.npy", "wb") as f:
                prb = pl_module.test_id.compute().cpu().numpy()
                np.save(f, prb)
        except:
            print("Data is not collected")

class CalculateRisk(pl.Callback):
    """Identify the uncertain region (for predictions)"""
    def on_validation_end(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trg = pl_module.val_trg.compute().cpu().numpy()
        prb = pl_module.val_prb.compute().cpu().numpy()
        risk = RiskControl(rstar=0.1, split = False)
        res = risk.run(preds=prb, targs=trg)
        trainer.logger.experiment.add_scalar(tag="risk/rstar", scalar_value = res["rstar"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/risk", scalar_value = res["risk"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/coverage", scalar_value = res["coverage"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/testrisk", scalar_value = res["testrisk"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/testcov", scalar_value = res["testcov"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/bound", scalar_value = res["bound"], global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(tag="risk/theta", scalar_value = res["theta"], global_step=trainer.current_epoch)

        pl_module.val_prb.reset()
        pl_module.val_trg.reset()
        pl_module.val_id.reset()


class ValidationPlot(pl.Callback):
    def on_train_epoch_end(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        num_targets=pl_module.hparams.num_targets
        
        if pl_module.hparams.loss_type in ["mse", "mae", "smooth"]:
            figure = self.make_joint(prb, trg)
        elif pl_module.hparams.loss_type == "corn":
            figure = self.make_figure(prb, trg)
        else:
            trg = pl_module.train_trg.compute().cpu().view(-1,4).numpy()
            prb = pl_module.train_prb.compute().view(-1,num_targets,5)
            for i in range(pl_module.hparams.num_targets):
                _trg = trg[:,i]
                _prb = prb[:,i].cpu().numpy()
                figure = self.make_figure(np.argmax(_prb, -1), _trg)
                trainer.logger.experiment.add_figure(tag="scores_train_%s" %i, figure = figure, global_step=trainer.current_epoch)

                figure = self.make_hist(_trg, "targets")
                trainer.logger.experiment.add_figure(tag="distr_trgs_%s" %i, figure = figure, global_step=trainer.current_epoch)
                plt.close()
        pl_module.train_prb.reset()
        pl_module.train_trg.reset()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        num_targets=pl_module.hparams.num_targets

        trg = pl_module.val_trg.compute().cpu().view(-1,num_targets).numpy()
        prb = pl_module.val_prb.compute().view(-1,num_targets,5)

        if True:
            for i in range(num_targets):
                _trg = trg[:,i]
                _prb = prb[:,i].cpu().numpy()
                figure = self.make_figure(np.argmax(_prb, -1), _trg)
                trainer.logger.experiment.add_figure(tag="scores_val_%s" %i, figure = figure, global_step=trainer.current_epoch)
                del figure
                plt.close()
        pl_module.val_prb.reset()
        pl_module.val_trg.reset()


    def make_figure(self, prb, trg):
        figure, ax = plt.subplots(1, figsize=(10,10))
        stats = confusion_matrix(trg,prb, normalize="true")
        sns.heatmap(stats, annot=True, ax=ax, cmap="Blues", vmin=0.0, vmax=1.0)
        return figure

    def make_hist(self,x, title):
        figure = plt.figure(figsize=(10,10))
        plt.title("Distribution of %s" %title)
        plt.hist(x, bins=40)
        plt.ylim([0,5000])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        return figure

    def make_joint(self, prb, trg):
        figure = plt.figure(figsize=(15,15))
        g = sns.jointplot(x=prb.reshape(-1), y=trg.reshape(-1), xlim=[0.,1.01], ylim=[0.,1.01], s=5,  marginal_kws=dict(bins=40))
        g.plot_joint(sns.kdeplot, color="g", zorder=0, levels=6, alpha=0.5, linewidth=0.5)
        g.ax_joint.set_xlabel("Predicted")
        g.ax_joint.set_ylabel("True")
        plt.tight_layout()

        return g.fig

    def make_plot(self, prb, trg):
        figure = plt.figure(figsize=(10,10))
        gs = GridSpec(4,4)
        plt.title("Probabilities vs Targets")
        ax_s = figure.add_subplot(gs[1:4, 0:3])
        ax_p = figure.add_subplot(gs[0, 0:3])
        ax_t = figure.add_subplot(gs[1:4,3])

        ax_s.scatter( prb, trg, s=5)
        ax_s.set_xlim([0.0,1])
        ax_s.set_ylim([0.0,1])

        ax_p.hist(prb, bins=40)
        ax_t.hist(trg, bins=40, orientation="horizontal")

        ax_t.set_xlim([0.0,1])
        ax_p.set_xlim([0.0,1])

        ax_s.set_xlabel("Predicted score")
        ax_t.set_ylabel("Target score")
        plt.tight_layout()
        return figure

class _ValidationPlot(pl.Callback):
    def on_train_epoch_end(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trg = pl_module.train_trg.compute().cpu().numpy()
        prb = pl_module.train_prb.compute().cpu().numpy()
        
        if pl_module.hparams.loss_type in ["mse", "mae", "smooth"]:
            figure = self.make_joint(prb, trg)
        elif pl_module.hparams.loss_type == "corn":
            figure = self.make_figure(prb, trg)
        else:
            trg = pl_module.train_trg.compute().long().cpu().numpy()
            prb = F.softmax(pl_module.train_prb.compute(), 1).cpu().numpy()
            figure = self.make_figure(np.argmax(prb, 1), trg)
        trainer.logger.experiment.add_figure(tag="scores_train", figure = figure, global_step=trainer.current_epoch)

        figure = self.make_hist(trg, "targets")
        trainer.logger.experiment.add_figure(tag="distr_trgs", figure = figure, global_step=trainer.current_epoch)

        pl_module.train_prb.reset()
        pl_module.train_trg.reset()
        plt.close()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    
        if pl_module.hparams.loss_type in ["mse", "mae", "smooth"]:
            trg = pl_module.val_trg.compute().cpu().numpy()
            prb = pl_module.val_prb.compute().cpu().numpy()
            figure = self.make_joint(prb, trg)
        elif pl_module.hparams.loss_type == "corn":
            trg = pl_module.val_trg.compute().long().cpu().numpy()
            prb = pl_module.val_prb.compute().cpu().numpy()
            figure = self.make_figure(prb, trg)
        else:
            trg = pl_module.val_trg.compute().long().cpu().numpy()
            prb = F.softmax(pl_module.val_prb.compute(), 1).cpu().numpy()
            figure = self.make_figure(np.argmax(prb, 1), trg)
        trainer.logger.experiment.add_figure(tag="scores", figure = figure, global_step=trainer.current_epoch)
        pl_module.val_prb.reset()
        pl_module.val_trg.reset()
        plt.close()


    def make_figure(self, prb, trg):
        figure, ax = plt.subplots(1, figsize=(10,10))
        stats = confusion_matrix(trg,prb, normalize="true")
        sns.heatmap(stats, annot=True, ax=ax, cmap="Blues", vmin=0.0, vmax=1.0)
        return figure

    def make_hist(self,x, title):
        figure = plt.figure(figsize=(10,10))
        plt.title("Distribution of %s" %title)
        plt.hist(x, bins=40)
        plt.ylim([0,5000])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        return figure

    def make_joint(self, prb, trg):
        figure = plt.figure(figsize=(15,15))
        g = sns.jointplot(x=prb.reshape(-1), y=trg.reshape(-1), xlim=[0.,1.01], ylim=[0.,1.01], s=5,  marginal_kws=dict(bins=40))
        g.plot_joint(sns.kdeplot, color="g", zorder=0, levels=6, alpha=0.5, linewidth=0.5)
        g.ax_joint.set_xlabel("Predicted")
        g.ax_joint.set_ylabel("True")
        plt.tight_layout()

        return g.fig

    def make_plot(self, prb, trg):
        figure = plt.figure(figsize=(10,10))
        gs = GridSpec(4,4)
        plt.title("Probabilities vs Targets")
        ax_s = figure.add_subplot(gs[1:4, 0:3])
        ax_p = figure.add_subplot(gs[0, 0:3])
        ax_t = figure.add_subplot(gs[1:4,3])

        ax_s.scatter( prb, trg, s=5)
        ax_s.set_xlim([0.0,1])
        ax_s.set_ylim([0.0,1])

        ax_p.hist(prb, bins=40)
        ax_t.hist(trg, bins=40, orientation="horizontal")

        ax_t.set_xlim([0.0,1])
        ax_p.set_xlim([0.0,1])

        ax_s.set_xlabel("Predicted score")
        ax_t.set_ylabel("Target score")
        plt.tight_layout()
        return figure

class EmbeddingCollector(pl.Callback):
    """Log the token embeddings on pre-training"""
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.hparams.parametrize_emb:
            w = pl_module.transformer.embedding.token.parametrizations.weight.original.detach()
        else:
            w = pl_module.transformer.embedding.token.weight.detach()
        trainer.logger.experiment.add_embedding(w, tag='token_embeddings', global_step=trainer.current_epoch)
        log.info("Token Embedding matrix is saved")
        return super().on_validation_end(trainer, pl_module)
    
class TextCollector(pl.Callback):
    """Collect the sequence data (original and predicted) during the MLM Task"""
    def __init__(self, num_samples_per_epoch: int = 3) -> None:
        super().__init__()
        self.num_samples_per_epoch = num_samples_per_epoch

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        self.collect(name = "train", trainer= trainer, pl_module = pl_module, batch = batch, batch_idx = batch_idx)
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)


    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx) -> None:
        self.collect(name = "val", trainer= trainer, pl_module = pl_module, batch = batch, batch_idx = batch_idx)
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)


    def collect(self, name: str, trainer, pl_module, batch, batch_idx,):
        IDX = 0
        #indx2token = pl_module.idx2token
        indx2token = trainer.datamodule.vocabulary.index2token
        if batch_idx < self.num_samples_per_epoch:
            

            original_sequence = batch["original_sequence"][IDX].long().tolist()
            abspos = batch["input_ids"][IDX][1].long().tolist()
            age = batch["input_ids"][IDX][2].long().tolist()
            masked_sequence = batch["input_ids"][IDX][0].long().tolist()
            predictions = torch.argmax(pl_module.forward(batch)[0][IDX], dim = 1).long().tolist()
            sequence_id = batch["sequence_id"][IDX].long().detach().tolist()
            target_pos= np.array(batch["target_pos"][IDX].long().tolist())

            try:
                end_of_sequence = original_sequence.index(0)
            except:
                end_of_sequence = len(original_sequence)

  
            trainer.logger.experiment.add_text(
                "%s/extended_sequences" %name,
                "(%s) " % sequence_id
                + " | ".join(
                    [
                        indx2token[token] + " (%s, %s) " % (abspos[i], age[i])
                        for i, token in enumerate(original_sequence[:end_of_sequence])
                    ]
                ),
                global_step=trainer.current_epoch,
            )

            trainer.logger.experiment.add_text(
                "%s/processed_sequences<O,M,P>" %name,
                "(%s) " % sequence_id +

                " | ".join(
                    [
                        indx2token[token]
                        if i not in target_pos
                        else "***"
                        + indx2token[token]
                        + "<>"
                        + indx2token[masked_sequence[i]] 
                        + ">>"
                        + indx2token[
                            predictions[np.where(i == target_pos)[0].tolist()[0]]
                    ]
                    + "***"
                        for i, token in enumerate(original_sequence[:end_of_sequence])
                    ]
                ),
                global_step=trainer.current_epoch,
            )
        
class TrackIDsEpoch(pl.Callback):
    """Tracks PERSON_IDs sampled per epoch"""
    def __init__(self) -> None:
        super().__init__()
        self.id_set = set()
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.id_set.clear()
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx) -> None:
        sequence_id = batch["sequence_id"].long().detach().tolist()
        if any(item in self.id_set for item in sequence_id):
            log.warning("Repeated Sequence IDs in the Validation Set")
        self.id_set.update(sequence_id)
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        train_ids = trainer.datamodule.corpus.population.data_split().train.tolist()
        for _id in batch["sequence_id"].long().detach().tolist():
            if _id in train_ids: 
                pass
            else: 
                log.info("Train ID %s is not part of the population" %int(_id))


class ReseedTrainDataLoader(pl.Callback):
    """reseed the trainloader before every epoch using PL_GLOBAL_SEED and epoch"""
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "PL_GLOBAL_SEED" not in os.environ:
            return

        if isinstance(trainer.train_dataloader.sampler, RandomSampler):
            pl_seed = int(os.environ["PL_GLOBAL_SEED"])
            epoch = trainer.current_epoch
            new_seed = get_hash({"seed": pl_seed, "epoch": epoch})
            log.info("Setting new dataloader seed= %d", new_seed)
            trainer.train_dataloader.sampler.generator.manual_seed(new_seed)

        ### QUESTION HERE
        elif isinstance(trainer.train_dataloader.sampler, WeightedRandomSampler):
            pl_seed = int(os.environ["PL_GLOBAL_SEED"])
            epoch = trainer.current_epoch
            new_seed = get_hash({"seed": pl_seed, "epoch": epoch})
            log.info("Setting new dataloader seed= %d", new_seed)
            trainer.train_dataloader.sampler.generator.manual_seed(new_seed)

    

# for single
class _RebalancedSampling(pl.Callback):
    def init_rebalancing(self, trainer, pl_module):
        self.n_classes = pl_module.hparams.num_classes
        self.n_samples = trainer.train_dataloader.sampler.weights.shape[0]
        self.n_epochs = trainer.max_epochs
        self._device = trainer.train_dataloader.sampler.weights.device
        self._dtype = trainer.train_dataloader.sampler.weights.dtype

        self.difficulty_matrix = torch.zeros((self.n_samples, self.n_epochs), 
                                              device=self._device, dtype=self._dtype)
        self.difficulty_matrix[:,0] = 1.
        self.prb_matrix = torch.zeros((self.n_samples, self.n_epochs + 1, self.n_classes),
                                       device=self._device, dtype=self._dtype)
        self.prb_matrix[:,0] = 1/self.n_classes
        self._w_epoch = torch.full(size = (self.n_epochs, 1), fill_value= 0.9,
                                    device=self._device, dtype=self._dtype)
        for i in range(1,self.n_epochs):
            self._w_epoch[i] = torch.pow(self._w_epoch[i], self.n_epochs-i)

        self.curr_diff = torch.ones(size=(self.n_samples, 1), device=self._device, dtype=self._dtype).reshape(-1)

        self.MAX_DIFF = 1000
        self.WIN_SIZE = 100


    def reweight_dataloader(self, trainer):
        epoch = trainer.current_epoch

        w = self.difficulty_matrix.sum(-1).to(self._device).type(self._dtype) 
        print(self.trgs[:4])
        print("sample 3")
        print("previous prb\t----------")
        print(self.prb_matrix[3, epoch])
        print("current prb\t------------")
        print(self.prb_matrix[3, epoch+1])
        print("difficulty matrix\t--------")
        print(self.difficulty_matrix[3,:10])
        print("sample 1")
        print("previous prb\t----------")
        print(self.prb_matrix[1, epoch])
        print("current prb\t------------")
        print(self.prb_matrix[1, epoch+1])
        print("difficulty matrix\t--------")
        print(self.difficulty_matrix[1,:10])

        q_min = torch.quantile(w, 0.1)
        q_max = torch.quantile(w, 0.9)
        w = torch.clip(w, min=q_min, max=q_max)
        trainer.train_dataloader.sampler.weights = w.to(self._device).type(self._dtype) 
        trainer.datamodule.weights = w


    def minmax_norm(self, x, eps: float = 1e-3):
        x = torch.nan_to_num(x, nan=eps)
        x_min, x_max = x.min(), x.max()
        return torch.clip((x-x_min)/(x_max - x_min), min=eps)


    def calculate_difficulty(self, epoch, eps: float = 1e-10):

        p_now = self.prb_matrix[:, epoch]
        p_lst = self.prb_matrix[:, epoch-1]
        diff = p_now - p_lst

        trgs = self.trgs.unsqueeze(1)

        du = torch.mul(torch.minimum(diff.gather(1,trgs), torch.zeros_like(diff.gather(1,trgs))),torch.log(torch.div(p_now.gather(1,trgs), p_lst.gather(1,trgs))))
        dl = torch.mul(torch.maximum(diff.gather(1,trgs), torch.zeros_like(diff.gather(1,trgs))), torch.log(torch.div(p_now.gather(1,trgs), p_lst.gather(1,trgs))))
        du = du.view(-1)
        dl = dl.view(-1)
        for i in range(diff.shape[0]):
            diff[i, self.trgs[i]] *=0

        du += torch.mul(torch.maximum(diff, torch.zeros_like(diff)),torch.log(torch.div(p_now, p_lst))).sum(-1)
        dl += torch.mul(torch.minimum(diff, torch.zeros_like(diff)),torch.log(torch.div(p_now, p_lst))).sum(-1)

        res =  torch.clip(torch.nan_to_num(torch.div(du, dl), nan=eps), min=eps, max=self.MAX_DIFF)
        return res
    


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if epoch == 0:
            self.init_rebalancing(trainer, pl_module)
        with torch.no_grad():
            softmax = SigSoftmax(dim=-1)
            pl_module.eval()
            prds = list()
            trgs = list()
            log.info("Calculating rebalanced weights...")
            for batch_idx, batch in enumerate(trainer.datamodule.rebalancing_dataloader()):
                for k, _ in batch.items():
                    batch[k] = batch[k].to(pl_module.device)
                logits = softmax(pl_module(batch))
                #if not torch.isclose(logits[0].sum(), torch.ones(1, device=pl_module.device)):
                #logits = torch.softmax(pl_module(batch), -1)
                prds.append(logits)
                trgs.append(pl_module.transform_targets(batch["target"], seq = batch["input_ids"], stage="train")[:,0])
            self.trgs = torch.cat(trgs).to(self._device).long()
            self.prb_matrix[:, epoch + 1] = torch.cat(prds, 0).to(self._device)
            self.difficulty_matrix[:, epoch + 1] = self.calculate_difficulty(epoch + 1)
            pl_module.train()
        self.reweight_dataloader(trainer)

class RebalancedSampling(pl.Callback):
    """Callback to rebalance samples based on the instance difficulty"""
    def init_rebalancing(self, trainer, pl_module):
        self.n_classes = pl_module.hparams.num_classes
        self.n_targets = 4
        self.n_samples = trainer.train_dataloader.sampler.weights.shape[0]
        self.n_epochs = trainer.max_epochs
        self._device = trainer.train_dataloader.sampler.weights.device
        self._dtype = trainer.train_dataloader.sampler.weights.dtype

        self.difficulty_matrix = torch.zeros((self.n_samples, self.n_epochs), 
                                              device=self._device, dtype=self._dtype)
        self.difficulty_matrix[:,0] = 1.
        self.prb_matrix = torch.zeros((self.n_samples, self.n_epochs + 1, self.n_targets, self.n_classes),
                                       device=self._device, dtype=self._dtype)
        self.prb_matrix[:,0] = 1/self.n_classes
        self._w_epoch = torch.full(size = (self.n_epochs, 1), fill_value= 0.9,
                                    device=self._device, dtype=self._dtype)
        for i in range(1,self.n_epochs):
            self._w_epoch[i] = torch.pow(self._w_epoch[i], self.n_epochs-i)

        self.curr_diff = torch.ones(size=(self.n_samples, 1), device=self._device, dtype=self._dtype).reshape(-1)

        self.MAX_DIFF = 100
        self.WIN_SIZE = 20


    def reweight_dataloader(self, trainer):
        """This method updates the dataloader associated with the Trainer (PyTorch Specific)"""
        epoch = trainer.current_epoch + 1
        decay = 0.5
        w = self.difficulty_matrix[:,0] 
        for i in range(epoch + 1):
            w = self.difficulty_matrix[:,i] * decay + (1-decay) * w
        w = self.robust_norm(w)
        trainer.train_dataloader.sampler.weights = w.to(self._device).type(self._dtype) 
        trainer.datamodule.weights = w

    def minmax_norm(self, x, eps: float = 1e-3):
        """MinMax Normlisation"""
        x = torch.nan_to_num(x, nan=eps)
        x_min, x_max = x.min(), x.max()
        return torch.clip((x-x_min)/(x_max - x_min), min=eps)

    def robust_norm(self, x, eps:float = 1e-5, q_min=0.25, q_max=0.75, with_centering: bool = False):
        """Robust Normalisation (aka Quantile-based)"""
        x = torch.nan_to_num(x, nan=eps)
        x_min = torch.quantile(x, q_min)
        x_max = torch.quantile(x, q_max)
        x_median = torch.median(x)
        if with_centering:
            x = (x-x_median)/(x_max-x_min)
        else:
            x = x/(x_max-x_min)
        return torch.nan_to_num(x, nan=eps)

    def calculate_difficulty(self, epoch, prb_matrix, trgs, eps: float = 1e-10):
        """Difficulty Calculations"""
        p_now = prb_matrix[:, epoch]
        p_lst = prb_matrix[:, epoch-1]
        diff = p_now - p_lst

        du = torch.mul(torch.minimum(diff.gather(1,trgs), torch.zeros_like(diff.gather(1,trgs))),torch.log(torch.div(p_now.gather(1,trgs), p_lst.gather(1,trgs))))
        dl = torch.mul(torch.maximum(diff.gather(1,trgs), torch.zeros_like(diff.gather(1,trgs))), torch.log(torch.div(p_now.gather(1,trgs), p_lst.gather(1,trgs))))
        du = du.view(-1)
        dl = dl.view(-1)
        for i in range(diff.shape[0]):
            diff[i, self.trgs[i]] *=0

        du += torch.mul(torch.maximum(diff, torch.zeros_like(diff)),torch.log(torch.div(p_now, p_lst))).sum(-1)
        dl += torch.mul(torch.minimum(diff, torch.zeros_like(diff)),torch.log(torch.div(p_now, p_lst))).sum(-1)

        res =  torch.clip(torch.nan_to_num(torch.div(du, dl), nan=eps), min=eps, max=self.MAX_DIFF)
        return res
    
    def calculate_agg_difficulty(self, epoch, eps: float = 1e-10):
        """If multiclass/multilabel prediction"""
        output = list()

        for i in range(0, self.n_targets):
            output.append(self.calculate_difficulty(epoch=epoch, prb_matrix=self.prb_matrix[:,:,i], 
                                            trgs=self.trgs[:,i].unsqueeze(1).long(),
                                            eps=eps))
        output = torch.stack(output)
        output = torch.amax(output, 0)
        return output


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        if epoch == 0:
            self.init_rebalancing(trainer, pl_module)
        with torch.no_grad():
            softmax = SigSoftmax(dim=-1)
            pl_module.eval()
            prds = list()
            trgs = list()
            log.info("Calculating rebalanced weights...")
            for batch_idx, batch in enumerate(trainer.datamodule.rebalancing_dataloader()):
                for k, _ in batch.items():
                    batch[k] = batch[k].to(pl_module.device)
                logits = softmax(pl_module(batch))
                prds.append(logits)
                trgs.append(pl_module.transform_targets(batch["target"], seq = batch["input_ids"], stage="train"))
            self.trgs = torch.vstack(trgs).to(self._device).long()
            self.prb_matrix[:, epoch + 1] = torch.cat(prds, 0).to(self._device)
            self.difficulty_matrix[:, epoch + 1] = self.robust_norm(
                                                        self.calculate_agg_difficulty(epoch + 1))
            pl_module.train()
        self.reweight_dataloader(trainer)