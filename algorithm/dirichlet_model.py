from typing import Any

import numpy as np
from scipy.stats import norm, entropy
import tensorflow as tf
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.base import TensorType, MeanAndVariance, InputData
from gpflow.models import GPModel
from gpflow import posteriors
from gpflow.kullback_leiblers import gauss_kl

from gpflow import logdensities
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

class DBModel(GPModel, InternalDataTrainingLossMixin):
    def tilde(self, data, a_eps):
        X, y = data
        # one-hot vector encoding
        Y01 = np.zeros((y.size, 2))
        Y01[:, 0], Y01[:, 1] = 1 - y, y
        s2_tilde = np.log(1.0 / (Y01 + a_eps) + 1)
        Y_tilde = np.log(Y01 + a_eps) - 0.5 * s2_tilde
        ymean = np.log(Y01.mean(0)) + np.mean(Y_tilde - np.log(Y01.mean(0)))
        Y_tilde = Y_tilde - ymean
        data_new = X, Y_tilde
        data_tilde = data_input_to_tensor(data_new)
        return data_tilde, s2_tilde

    def __init__(self, data, a_eps, Z, mean_function=None, **kwargs):
        self.data = data
        ## Parameterize a_eps and Z (wrap)
        self.a_eps = gpflow.Parameter(a_eps, trainable=True)
        self.Z = gpflow.Parameter(Z, trainable=False)
        self.inducing_variable = inducingpoint_wrapper(self.Z) ##? Unnecessary?
        ## Data transformation: set kernel
        self.data_tilde, self.s2_tilde = self.tilde(data, self.a_eps)
        X, Y_tilde = self.data_tilde
        kernel = gpflow.kernels.RBF(variance=np.var(Y_tilde), lengthscales=np.std(X))
        ## Set likelihood
        self.num_latent = Y_tilde.shape[1]
        likelihood = gpflow.likelihoods.MultiClass(num_classes=self.num_latent, invlink=None, **kwargs)

        super().__init__(
            kernel,
            likelihood,
            mean_function,
            num_latent_gps=Y_tilde.shape[-1],
            **kwargs
        )
        self.num_data = X.shape[0]
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

    def posterior(self):
        X, Y_tilde = self.data_tilde
        Z = self.inducing_variable

        kdiag = self.kernel(X, full_cov=True)
        s2 = tf.reshape(self.s2_tilde[:, 0], (X.shape[0], 1))
        K = kdiag + tf.linalg.diag(tf.squeeze(s2))
        kuf = Kuf(Z, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(L, Y_tilde, lower=True)
        alpha = tf.linalg.triangular_solve(tf.transpose(L), invL_y, lower=True)
        q_mu = tf.linalg.matmul(kuf, alpha)

        v = tf.linalg.triangular_solve(L, tf.transpose(kuf), lower=True)
        q_cov = kuu - tf.linalg.matmul(tf.transpose(v), v)
        return q_mu, q_cov

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self):
        return self.elbo()
    def elbo(self):
        X, Y = self.data
        q_mu, q_cov = self.posterior()
        q_sqrt = tf.linalg.diag(tf.sqrt(tf.linalg.diag_part(q_cov)))
        KL = gauss_kl(q_mu, q_sqrt, self.kernel(X, X))

        ## ?? Get expectations
        #var_exp = self.likelihood.variational_expectations(X, q_mu, q_cov, Y)

        log_softmax_q_mu = tf.nn.log_softmax(q_mu, axis=-1)
        L = tf.linalg.cholesky(q_cov)
        num_samples = 100
        standard_normal_samples = tf.random.normal(shape=[num_samples, *q_mu.shape])
        samples = q_mu + tf.matmul(standard_normal_samples, L, transpose_b=True)
        expectation = tf.reduce_mean(samples * tf.expand_dims(log_softmax_q_mu, axis=0), axis=0)

        return expectation - KL

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ): ## Should it use the same method as posterior?
        X, Y_tilde = self.data_tilde

        kdiag = self.kernel(X, full_cov=True)
        s2 = tf.reshape(self.s2_tilde[:, 0], (X.shape[0], 1))
        K = kdiag + tf.linalg.diag(tf.squeeze(s2))
        kuf = Kuf(Z, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(L, Y_tilde, lower=True)
        alpha = tf.linalg.triangular_solve(tf.transpose(L), invL_y, lower=True)
        fmu = tf.linalg.matmul(kuf, alpha)

        v = tf.linalg.triangular_solve(L, tf.transpose(kuf), lower=True)
        fcov = kuu - tf.linalg.matmul(tf.transpose(v), v)
        return fmu, fcov

