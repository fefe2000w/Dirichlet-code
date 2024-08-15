
import numpy as np


def get_split_data(data, name, split_index):
    x = data["x"]
    t = data["t"]
    print(name, split_index)
    train_idx = data["train"][split_index] - 1
    test_idx = data["test"][split_index] - 1
    X = x[train_idx.flatten(), :]
    y = t[train_idx.flatten()]
    Xtest = x[test_idx.flatten(), :]
    ytest = t[test_idx.flatten()]
    y = np.where(y == -1, 0, y)
    ytest = np.where(ytest == -1, 0, ytest)
    return X, y, Xtest, ytest