import gpflow
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import sys

sys.path.insert(0, "src")
import dirichlet_model


N = 20  # training data
np.random.seed(1235)


## create synthetic dataset
## ===============================
xmax = 15
X = np.random.rand(N, 1) * xmax
Xtest = np.linspace(0, xmax * 1.5, 200).reshape(-1, 1)
Z = X.copy()

y = np.cos(X.flatten()) / 2 + 0.5
y = np.random.rand(y.size) > y
y = y.astype(int)
if np.sum(y == 1) == 0:
    y[0] = 1
elif np.sum(y == 0) == 0:
    y[0] = 0


## GP setup and hyperparam optimisation
## ====================================
model = dirichlet_model.DBModel((X, y), a_eps=0.01, Z=Z)
# model.kernel.lengthscales = np.std(X)
# model.kernel.variance = np.var(Y_tilde)


opt = gpflow.optimizers.Scipy()
print("\nloglik (before) =", model.maximum_log_likelihood_objective())
# print('ampl =', model.kernel.variance)
# print('leng =', model.kernel.lengthscales)
print(model.trainable_parameters)
opt.minimize(model.training_loss, model.trainable_variables)
print("loglik  (after) =", model.maximum_log_likelihood_objective())
# print('ampl =', model.kernel.variance)
# print('leng =', model.kernel.lengthscales)
print(model.trainable_parameters)
