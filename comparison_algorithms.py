# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:35:47 2024

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
import evaluation

import scipy.io as scio
database = scio.loadmat("D:/A Sem1 2024/COMP8755/code/datasets/benchmarks.mat")
benchmarks = database['benchmarks'][0]

def get_split_data(data, name, split_index):
    x = data['x']
    t = data['t']
    print(name, split_index)
    train_idx = data['train'][split_index] - 1  
    test_idx = data['test'][split_index] - 1
    X = x[train_idx.flatten(), :]
    y = t[train_idx.flatten()]
    Xtest = x[test_idx.flatten(), :]
    ytest = t[test_idx.flatten()]
    y = np.where(y == -1, 0, y)
    ytest = np.where(ytest == -1, 0, ytest)
    return X, y, Xtest, ytest

all_report = {}
all_errors = {}
for benchmark in benchmarks:
    name = benchmark[0]
    if name == 'image' or name == 'splice':
        continue
    
    splits = 10
    data = database[name][0, 0]
    benchmark_reports = []
    
    for split_index in range(splits):
        X, y, Xtest, ytest = get_split_data(data, name, split_index)

        ARD = False
        report = {}
        report['ARD'] = ARD
        report['training_size'] = X.shape[0]
        report['test_size'] = Xtest.shape[0]

        ytest = ytest.astype(int)
        report['ytest'] = ytest
        
        Z = None
    

        vi_report = evaluation.evaluate_vi(X, y, Xtest, ytest, ARD=ARD, Z=Z)
        
        la_report = evaluation.evaluate_la(X, y, Xtest, ytest, ARD=ARD, Z=Z)    

        ep_report = evaluation.evaluate_ep(X, y, Xtest, ytest, ARD=ARD, Z=Z) 
        
        optimal_a_eps, optimization_time = evaluation.optimize_a_eps(X, y, Xtest, ytest, ARD=ARD, Z=Z)

        db_report = evaluation.evaluate_db(X, y, Xtest, ytest, optimal_a_eps, ARD=ARD, Z=Z)        
        db_report['db_elapsed_optim'] += optimization_time

        report = {**report, **vi_report, **db_report, **la_report, **ep_report}
        benchmark_reports.append(report)
    
    average_report = {}
    error_report = {}
    num_splits = len(benchmark_reports)
    keys = benchmark_reports[0].keys()

    for key in keys:
        if isinstance(benchmark_reports[0][key], (int, float, np.number)):
            values = [report[key] for report in benchmark_reports]
            average_report[key] = np.mean(values)
            error_report[key] = np.std(values) / np.sqrt(num_splits)  
        else:
            average_report[key] = benchmark_reports[0][key]
            error_report[key] = None

    all_report[name] = average_report
    all_errors[name] = error_report



import matplotlib.pyplot as plt
dataset_names = list(all_report.keys())

vi_error_rate = []
db_error_rate = []
la_error_rate = []
ep_error_rate = []

vi_mnll = []
db_mnll = []
la_mnll = []
ep_mnll = []

vi_ece = []
db_ece = []
la_ece = []
ep_ece = []

vi_time = []
db_time = []
la_time = []
ep_time = []

vi_optim_time = []
db_optim_time = []
la_optim_time = []
ep_optim_time = []

vi_pred_time = []
db_pred_time = []
la_pred_time = []
ep_pred_time = []



for dataset in dataset_names:
    vi_error_rate.append(all_report[dataset]['vi_error_rate'])
    db_error_rate.append(all_report[dataset]['db_error_rate'])
    la_error_rate.append(all_report[dataset]['la_error_rate'])
    ep_error_rate.append(all_report[dataset]['ep_error_rate'])

    vi_mnll.append(all_report[dataset]['vi_mnll'])
    db_mnll.append(all_report[dataset]['db_mnll'])
    la_mnll.append(all_report[dataset]['la_mnll'])
    ep_mnll.append(all_report[dataset]['ep_mnll'])

    vi_ece.append(all_report[dataset]['vi_ece'])
    db_ece.append(all_report[dataset]['db_ece'])
    la_ece.append(all_report[dataset]['la_ece'])
    ep_ece.append(all_report[dataset]['ep_ece'])
    
    vi_time.append(all_report[dataset]['vi_elapsed_optim']+all_report[dataset]['vi_elapsed_pred'])
    db_time.append(all_report[dataset]['db_elapsed_optim']+all_report[dataset]['db_elapsed_pred'])
    la_time.append(all_report[dataset]['la_elapsed_optim']+all_report[dataset]['la_elapsed_pred'])
    ep_time.append(all_report[dataset]['ep_elapsed_optim']+all_report[dataset]['ep_elapsed_pred'])

    vi_optim_time.append(all_report[dataset]['vi_elapsed_optim'])
    db_optim_time.append(all_report[dataset]['db_elapsed_optim'])
    la_optim_time.append(all_report[dataset]['la_elapsed_optim'])
    ep_optim_time.append(all_report[dataset]['ep_elapsed_optim'])

    vi_pred_time.append(all_report[dataset]['vi_elapsed_pred'])
    db_pred_time.append(all_report[dataset]['db_elapsed_pred'])
    la_pred_time.append(all_report[dataset]['la_elapsed_pred'])
    ep_pred_time.append(all_report[dataset]['ep_elapsed_pred'])
    
    
vi_error_rate_err = []
db_error_rate_err = []
la_error_rate_err = []
ep_error_rate_err = []

vi_mnll_err = []
db_mnll_err = []
la_mnll_err = []
ep_mnll_err = []

vi_ece_err = []
db_ece_err = []
la_ece_err = []
ep_ece_err = []

for dataset in dataset_names:
    vi_error_rate_err.append(all_errors[dataset]['vi_error_rate'])
    db_error_rate_err.append(all_errors[dataset]['db_error_rate'])
    la_error_rate_err.append(all_errors[dataset]['la_error_rate'])
    ep_error_rate_err.append(all_errors[dataset]['ep_error_rate'])

    vi_mnll_err.append(all_errors[dataset]['vi_mnll'])
    db_mnll_err.append(all_errors[dataset]['db_mnll'])
    la_mnll_err.append(all_errors[dataset]['la_mnll'])
    ep_mnll_err.append(all_errors[dataset]['ep_mnll'])

    vi_ece_err.append(all_errors[dataset]['vi_ece'])
    db_ece_err.append(all_errors[dataset]['db_ece'])
    la_ece_err.append(all_errors[dataset]['la_ece'])
    ep_ece_err.append(all_errors[dataset]['ep_ece'])


bar_width = 0.2
index = np.arange(len(dataset_names))

fig, ax = plt.subplots(3, 1, figsize=(12, 18))

hatches = ['', '//', '||', '++']

ax[0].bar(index, vi_error_rate, bar_width, yerr=vi_error_rate_err, label='Variational Inference', hatch=hatches[0])
ax[0].bar(index + bar_width, db_error_rate, bar_width, yerr=db_error_rate_err, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[0].bar(index + 2 * bar_width, la_error_rate, bar_width, yerr=la_error_rate_err, label='Laplace Approximation', hatch=hatches[2])
ax[0].bar(index + 3 * bar_width, ep_error_rate, bar_width, yerr=ep_error_rate_err, label='Expectation Propagation', hatch=hatches[3])
ax[0].set_xlabel('Dataset')
ax[0].set_ylabel('Error Rate')
ax[0].set_title('Error Rate by Dataset and Method')
ax[0].set_xticks(index + 1.5 * bar_width)
ax[0].set_xticklabels(dataset_names, rotation=45)
ax[0].legend()

ax[1].bar(index, vi_mnll, bar_width, yerr=vi_mnll_err, label='Variational Inference', hatch=hatches[0])
ax[1].bar(index + bar_width, db_mnll, bar_width, yerr=db_mnll_err, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[1].bar(index + 2 * bar_width, la_mnll, bar_width, yerr=la_mnll_err, label='Laplace Approximation', hatch=hatches[2])
ax[1].bar(index + 3 * bar_width, ep_mnll, bar_width, yerr=ep_mnll_err, label='Expectation Propagation', hatch=hatches[3])
ax[1].set_xlabel('Dataset')
ax[1].set_ylabel('MNLL')
ax[1].set_title('MNLL by Dataset and Method')
ax[1].set_xticks(index + 1.5 * bar_width)
ax[1].set_xticklabels(dataset_names, rotation=45)
ax[1].legend()

ax[2].bar(index, vi_ece, bar_width, yerr=vi_ece_err, label='Variational Inference', hatch=hatches[0])
ax[2].bar(index + bar_width, db_ece, bar_width, yerr=db_ece_err, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[2].bar(index + 2 * bar_width, la_ece, bar_width, yerr=la_ece_err, label='Laplace Approximation', hatch=hatches[2])
ax[2].bar(index + 3 * bar_width, ep_ece, bar_width, yerr=ep_ece_err, label='Expectation Propagation', hatch=hatches[3])
ax[2].set_xlabel('Dataset')
ax[2].set_ylabel('ECE')
ax[2].set_title('ECE by Dataset and Method')
ax[2].set_xticks(index + 1.5 * bar_width)
ax[2].set_xticklabels(dataset_names, rotation=45)
ax[2].legend()

plt.tight_layout()
plt.show()

# Comparison of time
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

ax[0].bar(index, vi_time, bar_width, label='Variational Inference', hatch=hatches[0])
ax[0].bar(index + bar_width, db_time, bar_width, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[0].bar(index + 2 * bar_width, la_time, bar_width, label='Laplace Approximation', hatch=hatches[2])
ax[0].bar(index + 3 * bar_width, ep_time, bar_width, label='Expectation Propagation', hatch=hatches[3])
ax[0].set_xlabel('Dataset')
ax[0].set_ylabel('Time')
ax[0].set_title('Time by Dataset and Method')
ax[0].set_xticks(index + 1.5 * bar_width)
ax[0].set_xticklabels(dataset_names, rotation=45)
ax[0].legend()

ax[1].bar(index, vi_optim_time, bar_width, label='Variational Inference', hatch=hatches[0])
ax[1].bar(index + bar_width, db_optim_time, bar_width, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[1].bar(index + 2 * bar_width, la_optim_time, bar_width, label='Laplace Approximation', hatch=hatches[2])
ax[1].bar(index + 3 * bar_width, ep_optim_time, bar_width, label='Expectation Propagation', hatch=hatches[3])
ax[1].set_xlabel('Dataset')
ax[1].set_ylabel('Optimisation Time')
ax[1].set_title('Optim Time by Dataset and Method')
ax[1].set_xticks(index + 1.5 * bar_width)
ax[1].set_xticklabels(dataset_names, rotation=45)
ax[1].legend()

ax[2].bar(index, vi_pred_time, bar_width, label='Variational Inference', hatch=hatches[0])
ax[2].bar(index + bar_width, db_pred_time, bar_width, label='Dirichlet-Based GPC', hatch=hatches[1])
ax[2].bar(index + 2 * bar_width, la_pred_time, bar_width, label='Laplace Approximation', hatch=hatches[2])
ax[2].bar(index + 3 * bar_width, ep_pred_time, bar_width, label='Expectation Propagation', hatch=hatches[3])
ax[2].set_xlabel('Dataset')
ax[2].set_ylabel('Prediction Time')
ax[2].set_title('Pred Time by Dataset and Method')
ax[2].set_xticks(index + 1.5 * bar_width)
ax[2].set_xticklabels(dataset_names, rotation=45)
ax[2].legend()

plt.show()


# Measurement by time
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].scatter(vi_time, vi_error_rate, color='r', marker='o', label='Variational Inference')
axs[0].scatter(db_time, db_error_rate, color='g', marker='x', label='Dirichlet-Based GPC')
axs[0].scatter(la_time, la_error_rate, color='b', marker='s', label='Laplace Approximation')
axs[0].scatter(ep_time, ep_error_rate, color='m', marker='d', label='Expectation Propagation')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Error Rate')
axs[0].set_title('Error Rate vs Time')
axs[0].legend()

axs[1].scatter(vi_time, vi_mnll, color='r', marker='o', label='Variational Inference')
axs[1].scatter(db_time, db_mnll, color='g', marker='x', label='Dirichlet-Based GPC')
axs[1].scatter(la_time, la_mnll, color='b', marker='s', label='Laplace Approximation')
axs[1].scatter(ep_time, ep_mnll, color='m', marker='d', label='Expectation Propagation')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('MNLL')
axs[1].set_title('MNLL vs Time')
axs[1].legend()

axs[2].scatter(vi_time, vi_ece, color='r', marker='o', label='Variational Inference')
axs[2].scatter(db_time, db_ece, color='g', marker='x', label='Dirichlet-Based GPC')
axs[2].scatter(la_time, la_ece, color='b', marker='s', label='Laplace Approximation')
axs[2].scatter(ep_time, ep_ece, color='m', marker='d', label='Expectation Propagation')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('ECE')
axs[2].set_title('ECE vs Time')
axs[2].legend()

plt.tight_layout()
plt.show()