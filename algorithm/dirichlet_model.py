from typing import Any

import numpy as np
from scipy.stats import norm, entropy
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.base import TensorType, MeanAndVariance, InputData, Transform
from gpflow.models import GPModel
from gpflow import posteriors
from gpflow.kullback_leiblers import gauss_kl

from gpflow import logdensities
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float, positive
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

import tensorflow_probability as tfp

# class SigmoidTransform(tfb.Bijector):
#     def __init__(self):
#         super().__init__(forward_min_event_ndims=0, inverse_min_event_ndims=0, name="sigmoid_transform")
#
#     def _forward(self, x):
#         return tf.math.sigmoid(x)
#
#     def _inverse(self, y):
#         return tf.math.log(y / (1 - y))
#
#     def _forward_log_det_jacobian(self, x):
#         return -tf.nn.softplus(-x) - tf.nn.softplus(x)
#
#     def _inverse_log_det_jacobian(self, y):
#         return -tf.math.log(y) - tf.math.log(1 - y)

class DBModel(GPModel, InternalDataTrainingLossMixin):
    def tilde(self, data, a_eps):
        a_eps = tf.exp(a_eps)
        X, y = data
        # one-hot vector encoding
        # Y01 = np.zeros((y.size, 2))
        # Y01[:, 0], Y01[:, 1] = 1 - y, y
        y_vec = y.astype(int)
        classes = np.max(y_vec).astype(int) + 1
        Y01 = np.zeros((len(y_vec), classes))
        for i in range(len(y_vec)):
            Y01[i, y_vec[i]] = 1
        s2_tilde = tf.math.log(1.0 / (Y01 + a_eps) + 1)
        Y_tilde = tf.math.log(Y01 + a_eps) - 0.5 * s2_tilde
        data_tilde = data_input_to_tensor((X, Y_tilde))
        return data_tilde, s2_tilde

    def __init__(self, data, a_eps, Z, mean_function=None, **kwargs):
        self.data = data
        ## Parameterize a_eps and Z (wrap)
        self.a_eps = gpflow.Parameter(a_eps, trainable=True)
        self.Z = gpflow.Parameter(Z, trainable=False)
        ## Data transformation: set kernel
        kernel = gpflow.kernels.RBF(
            variance=1.0, lengthscales=np.std(data[0], axis=0)
        )
        ## Set likelihood
        self.num_latent = num_latent = np.max(data[1]) + 1
        likelihood = gpflow.likelihoods.MultiClass(
            num_classes=self.num_latent, invlink=None, **kwargs
        )

        super().__init__(
            kernel,
            likelihood,
            mean_function,
            num_latent_gps=num_latent,
            **kwargs
        )
        # self.num_data = X.shape[0]
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

    def posterior(self):
        X, Y_tilde = self.data_tilde
        kff = self.kernel(X, full_cov=True)
        kff += tf.linalg.diag(
            default_jitter() * tf.ones(X.shape[0], dtype=default_float())
        )
        s2_tilde = self.s2_tilde.numpy()
        noise = tf.linalg.diag(tf.transpose(s2_tilde))
        K = kff + noise
        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(
            L, tf.expand_dims(tf.transpose(Y_tilde), -1), lower=True
        )
        invL_kff = tf.linalg.triangular_solve(L, kff, lower=True)
        q_mu = tf.squeeze(
            tf.linalg.matmul(tf.transpose(invL_kff, perm=[0, 2, 1]), invL_y)
        )
        q_cov = kff - tf.linalg.matmul(
            tf.transpose(invL_kff, perm=[0, 2, 1]), invL_kff
        )
        return q_mu, q_cov

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        self.data_tilde, self.s2_tilde = self.tilde(self.data, self.a_eps)
        X, Y = self.data
        q_mu, q_cov = self.posterior()
        q_sqrt = tf.linalg.cholesky(q_cov)
        kff = self.kernel(X, full_cov=True)
        kff += tf.linalg.diag(
            default_jitter() * tf.ones(X.shape[0], dtype=default_float())
        )
        KL = gauss_kl(tf.transpose(q_mu), q_sqrt, kff)

        f_mean = tf.transpose(q_mu)
        f_var = tf.transpose(tf.linalg.diag_part(q_cov))
        if Y.ndim == 1:
            var_exp = self.likelihood.variational_expectations(
                X, f_mean, f_var, np.expand_dims(Y, -1)
            )
        else:
            var_exp = self.likelihood.variational_expectations(
                X, f_mean, f_var, Y
            )
        expectation = tf.reduce_sum(var_exp)

        return expectation - KL

    @inherit_check_shapes
    def predict_f(
            self,
            Xnew: InputData,
            full_cov: bool = False,
            full_output_cov: bool = False,
    ):
        X, Y_tilde = self.data_tilde
        kxx = self.kernel(X)
        kxx += tf.linalg.diag(
            default_jitter() * tf.ones(X.shape[0], dtype=default_float())
        )
        s2_tilde = self.s2_tilde.numpy()
        noise = tf.linalg.diag(tf.transpose(s2_tilde))
        K = kxx + noise
        kxn = self.kernel(X, Xnew)
        knn = self.kernel(Xnew)

        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(
            L, tf.expand_dims(tf.transpose(Y_tilde), -1), lower=True
        )
        invL_knx = tf.linalg.triangular_solve(L, kxn, lower=True)
        fmu = tf.transpose(tf.squeeze(
            tf.linalg.matmul(tf.transpose(invL_knx, perm=[0, 2, 1]), invL_y)
        ))

        cov = knn - tf.linalg.matmul(
            tf.transpose(invL_knx, perm=[0, 2, 1]), invL_knx
        )
        fcov = tf.transpose(tf.linalg.diag_part(cov))
        return fmu, fcov
