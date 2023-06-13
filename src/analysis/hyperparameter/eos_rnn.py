import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer, Categorical
import pandas as pd
import glob 
import pickle
import os


### Script to run the hyperparameter search for RNN-based model.
### This script generates the values for the parameters
### You manually add the performance values.



N_TO_ASK = 6 # number of points to evaluate at the same time
N_TO_EVALUATE = 25 #number of points


def dump_results(optimizer, dir_path="../analysis/hyper/scikit_pretraining/", name="results_rnn.pkl"):
        if not(os.path.exists(dir_path)):
                os.mkdir(dir_path)
        with open(dir_path + name, "wb") as f:
                results = {"Xi": optimizer.Xi, 
                           "yi": optimizer.yi,
                           "optimizer": optimizer}
                pickle.dump(results, f)




search_space = [Integer(64, 768, name="hidden_size"),  
                Integer(1,8, name="n_layers"),
                Real(0.0, 0.5, name="dropout"), 
                Categorical([True, False], name="bidirectional")
                ]               



optimizer = Optimizer(dimensions = search_space,
                      base_estimator="GP",
                      acq_func="PI",
                      acq_optimizer="lbfgs",
                      random_state = 2021)



print("Points evaluated:", len(optimizer.Xi))
query = optimizer.ask(n_points=N_TO_ASK)
query_queue = []
for q in query:
        query_queue.append(q)
        print("\t", query_queue[-1])

while len(optimizer.Xi) <= N_TO_EVALUATE:

        if len(query_queue) == 0:
                query = optimizer.ask(n_points=N_TO_ASK)
                for q in query:
                    query_queue.append( q)
                    print("\t", query_queue[-1])


        current_querry = query_queue.pop(0)
        try:
                metrics = np.log(1. - float(input("AUL for the %s: " %current_querry )))
        except:
                print("Wrong Format. Skipping the query...")
                continue


        print("\tMetric: %.3f" %metrics)
        optimizer.tell(current_querry, metrics)
        print("Points evaluated:", len(optimizer.Xi))
        dump_results(optimizer)


i_best = np.argmin(optimizer.yi)
print("Best metric: %.3f" %optimizer.yi[i_best])
print("\t", optimizer.Xi[i_best])



