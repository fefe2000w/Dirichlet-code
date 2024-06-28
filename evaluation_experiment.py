import sys
import os
import pickle
import numpy as np
import tensorflow as tf
import gpflow
import GPy

sys.path.insert(0, "src")
import datasets
import evaluation


import scipy.io as scio

database = scio.loadmat("./datasets/benchmarks.mat")
benchmarks = database["benchmarks"][0]

def convert_to_serializable(obj):
    if isinstance(obj, gpflow.base.Parameter):
        return obj.numpy()  # Convert to numpy array
    elif isinstance(obj, tf.Tensor):
        return obj.numpy()  # Convert to numpy array
    elif isinstance(obj, GPy.core.parameterization.param.Param):
        return obj.values  # Convert to numpy array
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def experiment(method_name, method_evaluation, benchmarks, database):
    all_reports = {}
    all_errors = {}

    for benchmark in benchmarks:
        name = benchmark[0]
        if name == "image" or name == 'splice':
            continue

        splits = 10
        data = database[name][0, 0]
        benchmark_reports = []

        for split_index in range(splits):
            X, y, Xtest, ytest = datasets.get_split_data(data, name, split_index)

            ARD = False
            report = {}
            report["ARD"] = ARD
            report["training_size"] = X.shape[0]
            report["test_size"] = Xtest.shape[0]

            ytest = ytest.astype(int)
            report["ytest"] = ytest

            Z = None
            method_report = method_evaluation(X, y, Xtest, ytest, ARD, Z)

            report.update(method_report)
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

        all_reports[name] = convert_to_serializable(average_report)
        all_errors[name] = convert_to_serializable(error_report)

    result_dir = os.path.join('results', 'evaluation')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for benchmark_name in all_reports.keys():
        report_path = os.path.join(result_dir, f'{benchmark_name}_{method_name}_report.dat')
        error_path = os.path.join(result_dir, f'{benchmark_name}_{method_name}_errors.dat')

        with open(report_path, 'wb') as f:
            pickle.dump(all_reports[benchmark_name], f)
        with open(error_path, 'wb') as f:
            pickle.dump(all_errors[benchmark_name], f)


experiment('vi', evaluation.evaluate_vi, benchmarks, database)

experiment('la', evaluation.evaluate_la, benchmarks, database)

experiment('ep', evaluation.evaluate_ep, benchmarks, database)


def evaluate_db(X, y, Xtest, ytest, ARD, Z):
    optimal_a_eps, optimization_time = evaluation.optimize_a_eps(X, y, Xtest, ytest, ARD=ARD, Z=Z)
    db_report = evaluation.evaluate_db(X, y, Xtest, ytest, optimal_a_eps, ARD=ARD, Z=Z)
    db_report["db_elapsed_optim"] += optimization_time
    return db_report


experiment('db', evaluate_db, benchmarks, database)
