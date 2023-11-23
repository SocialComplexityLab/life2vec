from abc import ABC, abstractmethod
from ast import Str
from posixpath import split
from sys import meta_path
import numpy as np
import time
import os
import pickle
from typing import Any, List, Tuple, Callable, Dict, NamedTuple
from functools import reduce
import warnings
from torch.nn import Module
import torch

from captum.attr import LayerActivation, LayerGradientXActivation, LayerIntegratedGradients, IntegratedGradients, InputXGradient
from sklearn.base import BaseEstimator

from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer, matthews_corrcoef, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import BaggingClassifier

ATTR_PATH = r"../analysis/tcav/"

class BaseAttributions(ABC):
    """Base class for the Model Attribution"""
    @abstractmethod
    def get_concept_activations(self, calculate):
        NotImplemented

    def get_dataloader(self, dataset, indices):
        return self.data.get_fixed_dataloader(dataset, indices)

    def calculate_activations(self, data_loader):
        """Returns the activations for a each item after a given layer
        Args:
            data_iterator: Iterator that iterates over samples
        """
        layer_modules = [get_module_from_name(self.model, l) for l in self.layer] 
        layer_act = LayerActivation(self.model.forward_with_embeddings, layer_modules)
        activations  = list() 
        for batch in data_loader:
            embeddings, _ = self.model.transformer.get_sequence_embedding(batch["input_ids"].long().to(self.device))
            embeddings = embeddings.detach()
            act = layer_act.attribute(embeddings, additional_forward_args={"padding_mask": batch["padding_mask"].long().to(self.device)}, attribute_to_layer_input=True)
            activations.append(act[0].detach().cpu().numpy())

        return np.vstack(activations)



class Attributions(BaseAttributions):
    """Produces Attribution for random samples in TEST SET"""
    def __init__(self, 
                version: str,
                model: Tuple[Module, None],
                data,
                layers: List[str],
                sample_idx = None,
                custom_dataloader = None,
                random_seed: int = 2021):

        self.random_seed = random_seed
        self.version = version

        self.layer = layers
        self.model = model
        self.data = data
        
        assert len(layers) == 1

        if sample_idx is None:
            sample_idx = index_sample_data(self.data.test, sample_size=10000, random_seed=random_seed)

        if custom_dataloader is None:   
            self.loader_samples = self.get_dataloader(self.data.test, sample_idx)
        else: 
            self.loader_samples = custom_dataloader

        if self.model is None:
            warnings.warn("Model is not provided, use load_model method to assign a nn.Module")  
        else:
            self.device = model.device
            self.model.eval()

    def get_saliency(self, data, target: int, normalized: bool = True): 
        attr = InputXGradient(self.model.forward_with_embeddings)
        embeddings, _ = self.model.transformer.get_sequence_embedding(data["input_ids"].long().to(self.device))
        embeddings = embeddings.detach()
        embeddings.requires_grad = True
        self.model.zero_grad()
        out =  attr.attribute(embeddings,  target = target, additional_forward_args={"padding_mask": data["padding_mask"].long().to(self.device)})
        if normalized:
            output = []
            for i in range(0, data["input_ids"].shape[0]):
                output.append(self.summarize_attr(out[i], data["padding_mask"][i].bool().to(self.device)).detach())
            return output
        return out
    @staticmethod
    def summarize_attr(attr, mask):
        attr = attr.sum(dim=-1)
        attr = attr[mask]
        return attr/torch.norm(attr)
    def get_concept_activations(self, calculate: bool = False):
        output_folder = ATTR_PATH + "sample_act/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/act.pkl"
        if calculate:
            print("==== ACTIVATIONS START =====")
            activations = self.calculate_activations(self.loader_samples)
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump(activations, f)
            print("==== ACTIVATIONS CALCULATED =====")
            print("\t Results saved to", output_path)
        with open(output_path, "rb") as f:
            activations = pickle.load(f)
        return activations


    def get_concept_attributions(self, calculate: bool = False, attr_type: str = "integrated", multiply_by_inputs: bool = False, attribute_to_layer_input: bool = False):

        output_folder = ATTR_PATH + "sample_attr/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/attr.pkl" 
        if calculate:
            print("==== ACTTRIBUTIONS START =====")
            attr_pos = self.calculate_attributions(self.loader_samples, target = 1, multiply_by_inputs = multiply_by_inputs, attr_type= attr_type)
            print("\tPOS DONE")
            attr_neg = self.calculate_attributions(self.loader_samples, target = 0, multiply_by_inputs = multiply_by_inputs, attr_type = attr_type)
            print("\tNEG DONE")
            attributions = {"pos": attr_pos, "neg": attr_neg}
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump(attributions, f)
            print("==== ATTRIBUTIONS CALCULATED =====")
            print("\t Results saved to", output_path)
        with open(output_path, "rb") as f:
            attributions = pickle.load(f)
        return attributions
        
    def calculate_attributions(self, data_loader, target, attr_type: str = "simple", multiply_by_inputs: bool = False, attribute_to_layer_input: bool = False):
        assert attr_type in ["integrated", "simple"]
        layer_modules = [get_module_from_name(self.model, l) for l in self.layer] 
        if attr_type == "integrated":
            layer_attr = LayerIntegratedGradients(self.model.forward_with_embeddings,  layer_modules, multiply_by_inputs = multiply_by_inputs)
        elif attr_type == "simple":
            layer_attr = LayerGradientXActivation(self.model.forward_with_embeddings, layer_modules, multiply_by_inputs = multiply_by_inputs)
        else:
            raise NotImplementedError("ATTR_TYPE should be either 'simple' or 'intigrated'")
        attributions  = list() 
        for batch in data_loader:
            embeddings, _ = self.model.transformer.get_sequence_embedding(batch["input_ids"].long().to(self.device))
            embeddings = embeddings.detach()
            embeddings.requires_grad = True
            self.model.zero_grad()
            attr = layer_attr.attribute(embeddings, target = target,  
            additional_forward_args={"padding_mask": batch["padding_mask"].long().to(self.device)}, attribute_to_layer_input=attribute_to_layer_input)
            attributions.append(attr[0].detach().cpu().numpy())
            torch.cuda.empty_cache()
        return np.vstack(attributions)

    def get_concept_metadata(self, calculate):
        output_folder = ATTR_PATH + "sample_meta/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/meta.pkl" 
        if calculate:
            sequence = []
            sequence_ids = []
            targets = []
            preds = []
            print("==== METADATA START =====")
            for batch in self.loader_samples:

                for k in batch.keys():
                    torch.cuda.empty_cache()
                    batch[k] = batch[k].to(self.model.device)
                preds.append(self.model(batch).detach().cpu().numpy())
                sequence_ids.extend(batch["sequence_id"].tolist())
                sequence.append(batch["input_ids"][:,0].detach().cpu().numpy())
                targets.extend(batch["target"].tolist())
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump({"metadata": np.concatenate(sequence), "sequence_ids": sequence_ids, "targets": targets, "predictions": np.concatenate(preds)}, f)
            print("==== METADATA COLLECTED =====")
            print("\t Results saved to", output_path)
        with open(output_path, "rb") as f:
            metadata = pickle.load(f)
        return metadata


class TCAV(BaseAttributions):
    """Class to process and output TCAV for use defined concepts"""
    def __init__(self, 
                version: str,
                model: Tuple[Module, None],
                data,
                index_data,
                linear_cls: None,
                concept_name: str, 
                layers: List[str],
                local_path: str, 
                random_seed: int = 2021):

        assert len(layers) == 1, "TCAV is implemented only for one layer"

        self.random_seed = random_seed
        self.local_path = local_path 
        self.version = version
        self.concept_name = concept_name

        self.concept_name = concept_name.lower()
        self.layer = layers
        self.model = model
        self.data = data
        self.linear_cls = linear_cls

        self.loader_concepts = self.get_dataloader(self.data.val, index_data["concepts"])
        self.loader_randoms = self.get_dataloader(self.data.val, index_data["randoms"])

        if self.model is None:
            warnings.warn("Model is not provided, use load_model method to assign a nn.Module")  
        else:
            self.device = model.device
            self.model.eval()

        if self.linear_cls is None:
            warnings.warn("Linear Classifier is not provided. Use .set_classifier.")  

    def set_classifier(self, cls):
        self.linear_cls = cls
    
    def get_concept_activations(self, calculate: bool = False):
        output_folder = ATTR_PATH + "act/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/%s.pkl" %self.concept_name 
        if calculate:
            print("==== ACTIVATIONS START =====")
            act_concept = self.calculate_activations(self.loader_concepts)
            print("\tCONCEPTS DONE")
            act_random = self.calculate_activations(self.loader_randoms)
            print("\tRANDOMS DONE")
            activations = {"concepts" : act_concept, "randoms": act_random}
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump(activations, f)
            print("==== ACTIVATIONS CALCULATED =====")
            print("\tResults saved to", output_path)
        with open(output_path, "rb") as f:
            activations = pickle.load(f)
        return activations


    def get_cavs(self, calculate: bool = True,  n_bootstraps: int = 1000):
        output_folder = ATTR_PATH +"cavs/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/%s.pkl" %self.concept_name 

        if calculate:
            cavs = self.calculate_cavs(n_bootstraps = n_bootstraps)
            try:
                os.makedirs(output_folder, exist_ok=False)
            except:
                pass
            with open(output_path, "wb") as f:
                pickle.dump(cavs, f)
            print("==== CAVS CALCULATED =====")
            print("\tResults saved to", output_path)

        with open(output_path, "rb") as f:
            cavs = pickle.load(f)
        return cavs

    def calculate_score(self, attributions, cavs, magnitude: False):
        if magnitude:
            score = np.einsum("ij, bj -> ib", attributions, cavs) 
            return np.sum(np.abs(score * (score > 0)), axis=0)/(np.sum(np.abs(score), axis=0)+1e-12)
        return np.mean(np.einsum("ij, bj -> ib", attributions, cavs) > 0, axis= 0)


    def get_score(self, attributions, cavs, target, magnitude: bool = False):
        output_folder = ATTR_PATH + "score/%s_%s" %(self.version, self.layer[0])
        output_path = output_folder + "/%s_%s.pkl" %(self.concept_name, target)
        score = self.calculate_score(attributions, cavs, magnitude = magnitude)
        try:
            os.makedirs(output_folder, exist_ok=False)
        except:
            pass
        with open(output_path, "wb") as f:
            pickle.dump(score, f)
        print("==== SCORES CALCULATED =====")
        print("\tResults saved to", output_path)
        return score

    @staticmethod
    def load_score(version, layer, concept_name, target):
        output_folder = ATTR_PATH + "score/%s_%s" %(version, layer)
        output_path = output_folder + "/%s_%s.pkl" %(concept_name, target)
        with open(output_path, "rb") as f:
            score = pickle.load(f)
        return score

    def get_training_data(self):
        try:
            activations = self.get_concept_activations(calculate=False)
        except:
            activations = self.get_concept_activations(calculate=True)
        X_concept =  activations["concepts"]
        X_random = activations["randoms"]

        n_concept = X_concept.shape[0]
        n_random = X_random.shape[0]

        y_concept = np.ones(n_concept)
        y_random = np.zeros(n_random)

        X = np.concatenate([X_concept, X_random])
        y = np.concatenate([y_concept, y_random])

        return X,y

    def finetune_linear_cls(self, linear_cls, search_grid: Dict):
        X, y = self.get_training_data()
        scoring = make_scorer(matthews_corrcoef)
        search = GridSearchCV(linear_cls,
                                   search_grid, 
                                   cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0), scoring=scoring)
        search.fit(X, y)
        print("Searched Finished with MCC: %.2f" %search.best_score_)
        print("Best params:", search.best_params_)
        return search.best_params_


    def calculate_cavs(self, n_bootstraps, n_jobs: int = 10, model = None):
        X, y = self.get_training_data()
        if model is None: model = self.linear_cls
        m = BaggingClassifier(model, n_estimators=n_bootstraps, bootstrap=True, n_jobs=n_jobs, random_state = self.random_seed)
        m.fit(X,y)
        return np.vstack([c["model"].coef_ for c in m])
    

def index_sample_data(dataset, sample_size: int, random_seed: int=2021):
    np.random.seed(random_seed)
    dataset_size = len(dataset)
    return np.random.choice(np.arange(dataset_size), size=sample_size, replace=False)

def index_random_data(dataset, 
                    max_cpt_size: int = 3000,
                    max_rnd_size: int = 5000, 
                    random_seed: int = 2021):
    """Gives indices for random CONCEPT"""
    np.random.seed(random_seed)
    dataset_size = len(dataset)
    num_concepts = max_cpt_size + max_rnd_size
    idx = np.random.choice(np.arange(dataset_size), size=num_concepts, replace=False)
    output_path = ATTR_PATH + "index_data/random_data.pkl"
    with open(output_path, "wb") as f:
        out = {"concepts": set(idx[:max_cpt_size]), "randoms": set(idx[max_cpt_size:])}
        pickle.dump(out, f)
    print("==== INDEXING FINISHED ==========")
    print("\tResults saved to", output_path)
    print("=================================")
    return out

def index_concept_data(dataset, 
                       decision_fn, 
                       concept_name:str, 
                       max_cpt_size: int = 3000,
                       max_rnd_size: int = 5000,
                       random_seed: int = 2021
                       ):

    np.random.seed(random_seed)
    dataset_size = len(dataset)
    concepts = list()
    randoms = list()
    n_seen_concepts = 0
    n_seen_randoms = 0
    print("================================")
    print("==== INDEXING STARTED ==========")
    start_time = time.time()
    for i in range(dataset_size):
        if i%50000 == 1:
            print("%s out of %s (%.2f sec)" %(i, dataset_size, time.time()-start_time))
            print("\tConcept Size: %s Randoms Size: %s" %(len(concepts), len(randoms)))
        
        sample = dataset[i]
        if decision_fn(sample):
            n_seen_concepts += 1
            if len(concepts) < max_cpt_size:
                concepts.append(i)
            else:
                idx = np.random.randint(low=0, high = n_seen_concepts)
                if idx < max_cpt_size:
                    concepts[idx] = i
        else:
            n_seen_randoms += 1
            if len(randoms) < max_rnd_size:
                randoms.append(i)
            else:
                idx = np.random.randint(low=0, high = n_seen_randoms)
                if idx < max_rnd_size:
                    randoms[idx] = i 

    output_folder = ATTR_PATH + "index_data/"
    try:
        os.makedirs(output_folder, exist_ok=False)
    except:
        pass
    output_path = output_folder + "%s_data.pkl" %concept_name
    print("==== INDEXING FINISHED ==========")
    print("\tResults saved to", output_path)
    print("=================================")
    with open(output_path, "wb") as f:
        out = {"concepts": set(concepts), "randoms": set(randoms)}
        pickle.dump(out, f)
    return out


def get_module_from_name(model, layer_name: str) -> Any:
    r"""
    Returns the module (layer) object, given its (string) name
    in the model.
    Args:
        name (str): Module or nested modules name string in self.model

    Returns:
        The module (layer) in self.model.
    """
    return reduce(getattr, layer_name.split("."), model)