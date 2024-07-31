import numpy as np
from scipy.stats import norm, entropy
import tensorflow as tf
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.base import TensorType, MeanAndVariance
from gpflow.models import GPModel
from gpflow.likelihoods import ScalarLikelihood
from gpflow.posteriors import SGPRPosterior

from gpflow import logdensities
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

## Transform data
def tilde(data, a_eps):
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

def KL_divergence(mu1, sigma1, mu2, sigma2):
    inv_sigma2 = np.linalg.inv(sigma2)
    det_sigma1 = np.linalg.det(sigma1)
    det_sigma2 = np.linalg.det(sigma2)
    k = len(mu1)

    term1 = np.trace(inv_sigma2 @ sigma1)
    term2 = (mu2 - mu1).T @ inv_sigma2 @ (mu2 - mu1)
    term3 = -k
    term4 = np.log(det_sigma2 / det_sigma1)
    kl = 0.5 * (term1 + term2 + term3 + term4)

    return kl

def log_softmax(f):
    exp_f = np.exp(f)
    return f - np.log(np.sum(exp_f, axis=-1, keepdims=True))

class GaussianHeteroskedastic(ScalarLikelihood):
    def __init__(
        self, variance=1.0, scale=None, variance_lower_bound=None, **kwargs
    ):
        super().__init__(**kwargs)
        if np.isscalar(variance):
            variance = np.array(variance)
        self.variance = gpflow.Parameter(variance, trainable=False)
        # variance.trainable = False
        self.variance_numel = variance.size
        self.variance_ndim = variance.ndim

    @inherit_check_shapes
    def _scalar_log_prob(self, F, Y, X=None):
        return logdensities.gaussian(Y, F, self.variance)

    @inherit_check_shapes
    def _conditional_mean(self, F, X=None):
        return tf.idenity(F)

    @inherit_check_shapes
    def _conditional_variance(self, F, X=None):
        return tf.broadcast_to(self.variance, tf.shape(F))

    @inherit_check_shapes
    def _predict_mean_and_var(self, Fmu, Fvar, X=None):
        return tf.identity(Fmu), Fvar + self.variance

    @inherit_check_shapes
    def _predict_log_density(self, Fmu, Fvar, Y, X=None):
        return tf.reduce_sum(
            logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1
        )

    @inherit_check_shapes
    def _variational_expectations(self, Fmu, Fvar, Y, X=None):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance
        )

class Posterior(SGPRPosterior):
    def __init__(self, data, a_eps, Z, mean_function=None):
        data_tilde, s2_tilde = tilde(data, a_eps)
        X, Y = data_tilde
        self.variance = np.var(Y)
        self.s2_tilde = s2_tilde
        self.kernel = gpflow.kernels.RBF(variance=np.var(Y), lengthscales=np.std(X))
        likelihood = GaussianHeteroskedastic(s2_tilde)
        super().__init__(self.kernel, X, mean_function)
        self.Y_data = Y
        self.likelihood = likelihood
        self.Z = gpflow.Parameter(Z, trainable=False)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.inducing_variable: InducingPoints = inducingpoint_wrapper(self.Z)

    @inherit_check_shapes
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        mu = self.Y_data + mean_function
        cov = self.variance @ np.linalg.inv(self.variance + self.s2_tilde) @ self.s2_tilde
        return mu, cov

class SGPRh(GPModel, InternalDataTrainingLossMixin):
    def tilde(self, data, a_eps):
        X, y = data
        # one-hot vector encoding
        Y01 = np.zeros((y.size, 2))
        Y01[:, 0], Y01[:, 1] = 1 - y, y
        s2_tilde = np.log(1.0 / (Y01 + a_eps) + 1)
        Y_tilde = np.log(Y01 + a_eps) - 0.5 * s2_tilde
        ymean = np.log(Y01.mean(0)) + np.mean(Y_tilde - np.log(Y01.mean(0)))
        Y_tilde = Y_tilde - ymean
        data_tilde = data_input_to_tensor(X, Y_tilde)
        return data_tilde, s2_tilde

    def posterior(self):
        X, Y = self.data
        mu = Y + self.mean_function
        cov = self.variance @ np.linalg.inv(self.variance + self.s2_tilde) @ self.s2_tilde
        return mu, cov

    def __init__(self, data, a_eps, kernel, Z, mean_function=None, **kwargs):
        self.a_eps = gpflow.Parameter(a_eps, trainable=True)
        data_tilde, s2_tilde = tilde(data, self.a_eps)
        X, Y = data_tilde
        self.variance = np.var(Y)
        self.s2_tilde = s2_tilde
        self.kernel = gpflow.kernels.RBF(variance=np.var(Y), lengthscales=np.std(X))
        self.sn2 = s2_tilde
        likelihood = GaussianHeteroskedastic(self.sn2)
        super().__init__(
            kernel,
            likelihood,
            mean_function,
            num_latent_gps=Y.shape[-1],
            **kwargs
        )
        self.data = X, Y
        self.Z = gpflow.Parameter(Z, trainable=False)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.inducing_variable: InducingPoints = inducingpoint_wrapper(self.Z)
        self.mean_function = mean_function or gpflow.mean_functions.Zero()

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        mu, cov = self.posterior()
        KL = KL_divergence(self.mean_function, self.variance, mu, cov)

        num_samples = 10000
        mu1, mu2 = mu
        sigma1, sigma2 = cov
        f1_samples = np.random.normal(mu1, sigma1, num_samples)
        f2_samples = np.random.normal(mu2, sigma2, num_samples)

        f_samples = np.vstack([f1_samples, f2_samples]).T
        log_softmax_values = log_softmax(f_samples)
        expectation = np.mean(log_softmax_values[:,0] + log_softmax_values[:,1])

        bound = -KL + expectation
        return bound



    @inherit_check_shapes
    def predict_f(
            self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        X, Y = self.data
        Z = self.inducing_variable
        num_inducing = self.inducing_variable.num_inducing
        kuf = Kuf(Z, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        kus = Kuf(Z, self.kernel, Xnew)
        L = tf.linalg.cholesky(kuu)
        invL_kuf = tf.linalg.triangular_solve(L, kuf, lower=True)
        Err = Y - self.mean_function(X)

        mu = None
        cov = None
        for i in range(self.num_latent):
            err = tf.slice(Err, [0, i], [self.num_data, 1])

            if self.likelihood.variance_ndim > 1:
                sn2 = self.likelihood.variance[:, i]
            else:
                sn2 = self.likelihood.variance
            sigma = tf.sqrt(sn2)

            A = invL_kuf / sigma
            AAT = tf.linalg.matmul(A, A, transpose_b=True)
            B = AAT + tf.eye(num_inducing, dtype=default_float())
            LB = tf.linalg.cholesky(B)
            err_sigma = tf.reshape(err, [self.num_data]) / sigma
            err_sigma = tf.reshape(err_sigma, [self.num_data, 1])
            Aerr = tf.linalg.matmul(A, err_sigma)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
            tmp1 = tf.linalg.triangular_solve(L, kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)

            if full_cov:
                raise Exception("full_cov not imploemented!")
            else:
                var = (
                        self.kernel(Xnew, full_cov=False)
                        + tf.reduce_sum(tf.square(tmp2), 0)
                        - tf.reduce_sum(tf.square(tmp1), 0)
                )
                shape = tf.stack([1, tf.shape(err)[1]])
                var = tf.tile(tf.expand_dims(var, 1), shape)

            if mu is None or cov is None:
                mu = mean
                cov = var
            else:
                mu = tf.concat([mu, mean], 1)
                cov = tf.concat([cov, var], 1)

        mu = mu + self.mean_function(Xnew)
        return mu, cov
