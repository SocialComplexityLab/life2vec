import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer
import pandas as pd
import glob 
import pickle
import os

N_TO_ASK = 6
N_TO_EVALUATE = 25

def adjust_value(val:int,  d: int = 8, min_val: int = 0) -> int:
        remainder = val % d
        if val - remainder < min_val:
                val = val + d
                remainder = val % d
                return val - remainder
        return val - remainder

def dump_results(optimizer, dir_path="../analysis/hyper/scikit_pretraining/", name="results2.pkl"):
        if not(os.path.exists(dir_path)):
                os.mkdir(dir_path)
        with open(dir_path + name, "wb") as f:
                results = {"Xi": optimizer.Xi, 
                           "yi": optimizer.yi,
                           "optimizer": optimizer}
                pickle.dump(results, f)

def adjust_ask(asked, keys):
        config = {}
        for i in range(len(asked)):
                config[keys[i]] = asked[i]
        config["hidden_size"] = adjust_value(config["hidden_size"], config["n_heads"], min_val=64)

        config["n_local"] = np.floor(config["n_heads"] * config["n_local"])
        if config["n_local"] == 0:
            config["local_window"] = 0
        return config

def adjust_tell(config):
        """Convert n_local back to the ratio"""
        config["n_local"] /= config["n_heads"]
        return [v for k,v in config.items()]


params = [[336, 14, 14, 2/14, 2096,  64, 256], 
          [336, 14, 14, 2/14, 1785, 105,   4],
          [270,  6, 10, 1/10, 1964, 275,  99],
          [253,  6, 11, 9/11, 2439, 133, 120],
          [286,  8, 11, 6/11, 1213, 346,  55],
          [280,  5, 10, 7/10, 2210, 436,  93]]
scores = [np.log(1-i) for i in [0.326, 0.327, 0.328, 0.325, 0.325]]
space_keys = ["hidden_size", "n_encoders", "n_heads", "n_local", "hidden_ff", "n_rand_features", "local_window"]

search_space = [Integer(64, 336, name="hidden_size"),  
                Integer(4,14, name="n_encoders"),
                Integer(3,14, name="n_heads"), 
                Real(0.0, 1.0, name="n_local"), 
                Integer(512,2560, name="hidden_ff"), 
                Integer(64,512, name="num_random_features"),
                Integer(4,256, name="local_window_size"),
                ]               



optimizer = Optimizer(dimensions = search_space,
                      base_estimator="GP",
                      acq_func="PI",
                      acq_optimizer="lbfgs",
                      n_initial_points = 12,
                      random_state = 2021)

for i in range(len(scores)):
        optimizer.tell(params[i], scores[i])



print("Points evaluated:", len(optimizer.Xi))
query = optimizer.ask(n_points=N_TO_ASK)
query_adjusted = []

for i, q in enumerate(query):
        query_adjusted.append(adjust_ask(q, keys=space_keys))
        print("\t", query_adjusted[-1])

while len(optimizer.Xi) <= N_TO_EVALUATE:

        if len(query_adjusted) == 0:
                query = optimizer.ask(n_points=N_TO_ASK)
                for i, q in enumerate(query):
                        query_adjusted.append(adjust_ask(q, keys=space_keys))
                        print("\t", query_adjusted[-1])

        current_querry = query_adjusted.pop(0)
        try:
                metrics = np.log(1. - float(input("Weighted accuracy for the %s: " %current_querry )))
        except:
                print("Wrong Format. Skipping the query...")
                continue


        print("\tMetric: %.3f" %metrics)
        optimizer.tell(adjust_tell(current_querry), metrics)
        print("Points evaluated:", len(optimizer.Xi))
        dump_results(optimizer)


i_best = np.argmin(optimizer.yi)
print("Best metric: %.3f" %optimizer.yi[i_best])
print("\t", optimizer.Xi[i_best])



