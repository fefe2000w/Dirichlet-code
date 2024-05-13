# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:27:04 2024

@author: lisy
"""
import tensorflow as tf
import gpflow

import os
import sys
import time
import pickle
import random
import numpy as np
from scipy.cluster.vq import kmeans

sys.path.insert(0,'src')
import datasets
import evaluation
import utilities



## binary datasets: EEG, HTRU2, MAGIC
dataset = 'EEG'
split_idx = '01'
## cmd arguments to specify:
## dataset (optional) 
## split_idx (optional)

if len(sys.argv) > 1:
    dataset = str(sys.argv[1])
if len(sys.argv) > 2:
    split_idx = str(sys.argv[2])

if split_idx == '':
    split_idx = '01'
    print('Default split_idx: 01')
if dataset == '':
    print('')
    print('Script that performs evaluation of the following algorithms:')
    print(' - LA')
    print(' - EP')
    print(' - VI')
    print(' - DB')
    print('')
    exit()




ARD = False
subset = None
num_inducing = utilities.get_option_inducing(dataset)
a_eps = utilities.get_option_alphaEpsilon(dataset)

use_kmeans = True
test_subset = 20000
GPC_SKIP_THRESHOLD = 1000000
KMEANS_SKIP_THRESHOLD = 500000



path = utilities.get_dataset_path(dataset)
X, y, Xtest, ytest = datasets.load_split(path, split_idx)
X, Xtest = datasets.normalise_oneminusone(X, Xtest)

## create synthetic dataset
## ===============================
# N = 20  # training data
# np.random.seed(1235)
# xmax = 15
# X = np.random.rand(N,1) * xmax
# Xtest = np.linspace(0, xmax*1.5, 200).reshape(-1, 1)
# Z = X.copy()

# y = np.cos(X.flatten()) / 2 + 0.5
# y = np.random.rand(y.size) > y
# y = y.astype(int)
# if np.sum(y==1) == 0:
#     y[0] = 1
# elif np.sum(y==0) == 0:
#     y[0] = 0


# ytest = np.cos(Xtest.flatten()) / 2 + 0.5
# ytest = np.random.rand(ytest.size) > ytest
# ytest = ytest.astype(int)
# if np.sum(ytest==1) == 0:
#     ytest[0] = 1
# elif np.sum(ytest==0) == 0:
#     ytest[0] = 0
    
# X, Xtest = datasets.normalise_oneminusone(X, Xtest)   
if subset is not None:
    X = X[:subset, :]
    y = y[:subset]
if Xtest.shape[0] > test_subset:
    Xtest = Xtest[:test_subset, :]
    ytest = ytest[:test_subset]    





report = {}
report['ARD'] = ARD
report['training_size'] = X.shape[0]
report['test_size'] = Xtest.shape[0]
report['num_inducing'] = num_inducing
print('training_size =', X.shape[0])
print('test_size =', Xtest.shape[0])
print('num_inducing =', num_inducing, flush=True)

ytest = ytest.astype(int)
report['ytest'] = ytest

Z = None
if num_inducing is not None:
    if use_kmeans and X.shape[0] <= KMEANS_SKIP_THRESHOLD:
        print('kmeans... ', end='', flush=True)
        start_time = time.time()
        # kmeans returns a tuple
        Z = kmeans(X, num_inducing)[0]
        kmeans_elapsed = time.time() - start_time
        print('done!')
        report['kmeans_elapsed'] = kmeans_elapsed
        print('kmeans_elapsed =', kmeans_elapsed)
    else:
        shuffled = list(range(X.shape[0]))
        random.shuffle(shuffled)
        idx = shuffled[:num_inducing]
        Z = X[idx, :].copy()



'''
Evaluation of the following algorithms:
 - LA: Laplace approximation
 - EP: Expectation propagation
 - VI: Variational inference
 - DB: Dirichlet-based Gaussian process
'''

if X.shape[0] <= GPC_SKIP_THRESHOLD:
    vi_report = evaluation.evaluate_vi(X, y, Xtest, ytest, ARD=ARD, Z=Z)
else:
    print('GPC skipped: N >', GPC_SKIP_THRESHOLD)

db_report = evaluation.evaluate_db(X, y, Xtest, ytest, ARD=ARD, Z=Z, a_eps=0.1)        

la_report = evaluation.evaluate_la(X, y, Xtest, ytest, ARD=ARD, Z=Z)    

ep_report = evaluation.evaluate_ep(X, y, Xtest, ytest, ARD=ARD, Z=Z)    

report = {**report, **vi_report, **db_report, **la_report, **ep_report}


###############################################################################
# Save results
###############################################################################
# save_results = True
# result_dir = os.path.join('results', 'evaluation')

# if ARD:
#     result_path = os.path.join(result_dir, dataset+'_ard_report')
# else:
#     result_path = os.path.join(result_dir, dataset+'_iso_report')

# if save_results:
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)
#     result_dat = result_path + split_idx + '.dat'
#     pickle.dump(report, open(result_dat, 'wb'))
