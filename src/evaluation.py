# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:29:04 2024

@author: lisy
"""

import tensorflow as tf
import gpflow
import GPy as gp

import time
import numpy as np
from heteroskedastic import SGPRh


def nll(p, y):
    y = y.astype(int).flatten()
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.flatten()
        P = np.zeros([y.size, 2])
        P[:,0], P[:,1] = 1-p, p
        p = P
    classes = p.shape[1]
    Y = np.zeros((y.size, classes))
    for i in range(y.size):
        Y[i,y[i]] = 1
    logp = np.log(p)
    logp[np.isinf(logp)] = -750
    loglik = np.sum(Y * logp, 1)
    return -np.sum(loglik)


def calibration_test(p, y, nbins=10):
    '''
    Returns ece:  Expected Calibration Error
            conf: confindence levels (as many as nbins)
            accu: accuracy for a certain confidence level
                  We are interested in the plot confidence vs accuracy
            bin_sizes: how many points lie within a certain confidence level
    '''
    edges = np.linspace(0, 1, nbins+1)
    accu = np.zeros(nbins)
    conf = np.zeros(nbins)
    bin_sizes = np.zeros(nbins)
    # Multiclass problems are treated by considering the max
    if p.ndim>1 and p.shape[1]!=1:
        pred = np.argmax(p, axis=1)
        p = np.max(p, axis=1)
    else:
        # the treatment for binary classification
        pred = np.ones(p.size)
    #
    y = y.flatten()
    p = p.flatten()
    for i in range(nbins):
        idx_in_bin = (p > edges[i]) & (p <= edges[i+1])
        bin_sizes[i] = max(sum(idx_in_bin), 1)
        accu[i] = np.sum(y[idx_in_bin] == pred[idx_in_bin]) / bin_sizes[i]
        conf[i] = (edges[i+1] + edges[i]) / 2
    ece = np.sum(np.abs(accu - conf) * bin_sizes) / np.sum(bin_sizes)
    return ece, conf, accu, bin_sizes


################################################################################
### Classification utilising Variational Inference

# def evaluate_vi(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None):
#     report = {}
#     dim = X.shape[1]
#     if ARD:
#         default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
#     else:
#         default_len = np.mean(np.std(X,0))*np.sqrt(dim)

#     Y = y.reshape(y.size, 1)
#     classes = np.max(y).astype(int) + 1 

#     rbf_kern = gpflow.kernels.RBF(lengthscales=default_len, variance=np.var(Y))
#     variance = tf.Variable(1e-4, trainable=False)
#     white = gpflow.kernels.White(variance=variance)
#     kernel = rbf_kern + white
#     if classes == 2:
#         lik = gpflow.likelihoods.Bernoulli()
#     else:
#         lik = gpflow.likelihoods.MultiClass(classes)


#     if Z is None:
#         vi_model = gpflow.models.VGP((X, Y), kernel=kernel, likelihood=lik, 
#             num_latent_gps=classes)
#     else:
#         vi_model = gpflow.models.SVGP(kernel=kernel, likelihood=lik, 
#             inducing_variable=Z, num_latent_gps=classes, q_diag=False, whiten=True)
#         vi_model.inducing_variable.trainable = False
#     opt = gpflow.optimizers.Scipy()


#     if ampl is not None:
#         rbf_kern.variance.trainable = False
#         rbf_kern.variance = ampl * ampl
#     if leng is not None:
#         rbf_kern.lengthscales.trainable = False
#         if ARD:
#             rbf_kern.lengthscales = np.ones(dim) * leng
#         else:
#             rbf_kern.lengthscales = leng


#     print('vi optim... ', end='', flush=True)
#     start_time = time.time()    
#     opt.minimize(vi_model.training_loss, vi_model.trainable_variables)    
#     vi_elapsed_optim = time.time() - start_time
#     print('done!')
#     report['vi_elapsed_optim'] = vi_elapsed_optim


#     vi_amp = np.sqrt(rbf_kern.variance)
#     report['vi_amp'] = vi_amp
#     vi_len = rbf_kern.lengthscales
#     report['vi_len'] = vi_len


#     # predict
#     print('vi pred... ', end='', flush=True)
#     start_time = time.time()
#     vi_fmu, vi_fs2 = vi_model.predict_f(Xtest)
#     vi_prob = np.zeros(vi_fmu.shape)
#     #source = np.random.randn(1000, classes)
#     for i in range(vi_fmu.shape[0]):
#         samples = np.random.normal(vi_fmu[i,:], np.sqrt(vi_fs2[i,:]))
#         vi_prob[i,:] = samples.mean(0)
    
    
#     vi_elapsed_pred = time.time() - start_time
#     print('done!')
#     report['vi_elapsed_pred'] = vi_elapsed_pred



#     # also get this for reference
#     if classes == 2:
#         vi_pred = vi_prob > 0.5
#         vi_pred = vi_pred.astype(int).flatten()
#     else:
#         vi_pred = np.argmax(vi_prob, 1)

#     report['vi_pred'] = vi_pred
#     report['vi_prob'] = vi_prob
#     report['vi_fmu'] = vi_fmu
#     report['vi_fs2'] = vi_fs2


#     ytest = ytest.astype(int).flatten()
#     vi_error_rate = np.mean(vi_pred!=ytest)
#     report['vi_error_rate'] = vi_error_rate

#     vi_ece, conf, accu, bsizes = calibration_test(vi_prob, ytest)
#     report['vi_ece'] = vi_ece
#     vi_calib = {}
#     vi_calib['conf'] = conf
#     vi_calib['accu'] = accu
#     vi_calib['bsizes'] = bsizes
#     report['vi_calib'] = vi_calib

#     vi_nll = nll(vi_prob, ytest)
#     report['vi_nll'] = vi_nll

#     if classes == 2:
#         vi_typeIerror = np.mean(vi_pred[ytest==0])
#         report['vi_typeIerror'] = vi_typeIerror
#         vi_typeIIerror = np.mean(1-vi_pred[ytest==1])
#         report['vi_typeIIerror'] = vi_typeIIerror

#     print('vi_elapsed_optim =', vi_elapsed_optim)
#     print('vi_elapsed_pred =', vi_elapsed_pred)
#     print('---')
#     print('vi_amp =', vi_amp)
#     print('vi_len =', vi_len)
#     print('---')
#     print('vi_error_rate =', vi_error_rate)
#     if classes is None:
#         print('vi_typeIerror =', vi_typeIerror)
#         print('vi_typeIIerror =', vi_typeIIerror)
#     print('vi_ece =', vi_ece)
#     print('vi_nll =', vi_nll)
#     print('\n')
#     return report



################################################################################
### Classification with GPR on the transformed Dirichlet distribution

def evaluate_db(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None,
    a_eps=0.001, scale_sn2=False):

    report = {}
    dim = X.shape[1]
    if ARD:
        len0 = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        len0 = np.mean(np.std(X,0))*np.sqrt(dim)

    # prepare y: one-hot encoding
    y_vec = y.astype(int)
    classes = np.max(y_vec).astype(int) + 1
    Y = np.zeros((len(y_vec), classes))
    for i in range(len(y_vec)):
        Y[i, y_vec[i]] = 1



    # label transformation
    s2_tilde = np.log(1.0/(Y+a_eps) + 1)
    Y_tilde = np.log(Y+a_eps) - 0.5 * s2_tilde

    # For each y, we have two possibilities: 0+alpha and 1+alpha
    # Changing alpha (the scale of Gamma) changes the distance
    # between different class instances.
    # Changing beta (the rate of Gamma) changes the position 
    # (i.e. log(alpha)-log(beta)-s2_tilde/2 ) but NOT the distance.
    # Thus, we can simply move y for all classes to our convenience (ie zero mean)

    # 1st term: guarantees that the prior class probabilities are correct
    # 2nd term: just makes the latent processes zero-mean
    ymean = np.log(Y.mean(0)) + np.mean(Y_tilde-np.log(Y.mean(0)))
    Y_tilde = Y_tilde - ymean


    # set up regression
    var0 = np.var(Y_tilde)
    kernel = gpflow.kernels.RBF(lengthscales=len0, variance=var0)
    if Z is None:
        Z=X
    db_model = SGPRh((X, Y_tilde), kernel=kernel, sn2=s2_tilde, Z=Z)


    opt = gpflow.optimizers.Scipy()
    if ampl is not None:
        kernel.variance.trainable = False
        kernel.variance = ampl * ampl
    if leng is not None:
        kernel.lengthscales.trainable = False
        if ARD:
            kernel.lengthscales = np.ones(dim) * leng
        else:
            kernel.lengthscales = leng

    db_elapsed_optim = None
    if ampl is None or leng is None or a_eps is None:
        print('db optim... ', end='', flush=True)
        start_time = time.time()
        opt.minimize(db_model.training_loss, db_model.trainable_variables)
        db_elapsed_optim = time.time() - start_time
        print('done!')
        report['db_elapsed_optim'] = db_elapsed_optim


    db_amp = np.sqrt(db_model.kernel.variance)
    report['db_amp'] = db_amp
    db_len = db_model.kernel.lengthscales
    report['db_len'] = db_len
    report['db_a_eps'] = a_eps


    # predict
    print('db pred... ', end='', flush=True)
    start_time = time.time()
    db_fmu, db_fs2 = db_model.predict_f(Xtest)
    db_fmu = db_fmu + ymean

    # Estimate mean of the Dirichlet distribution through sampling
    db_prob = np.zeros(db_fmu.shape)
    source = np.random.randn(1000, classes)
    for i in range(db_fmu.shape[0]):
        samples = source * np.sqrt(db_fs2[i,:]) + db_fmu[i,:]
        samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
        db_prob[i,:] = samples.mean(0)

    db_elapsed_pred = time.time() - start_time
    print('done!')
    report['db_elapsed_pred'] = db_elapsed_pred



    # the actual prediction
    db_pred = np.argmax(db_prob, 1)
 
    report['db_pred'] = db_pred
    report['db_prob'] = db_prob
    report['db_fmu'] = db_fmu
    report['db_fs2'] = db_fs2


    db_error_rate = np.mean(db_pred!=ytest)
    report['db_error_rate'] = db_error_rate

    db_ece, conf, accu, bsizes = calibration_test(db_prob, ytest)
    report['db_ece'] = db_ece
    db_calib = {}
    db_calib['conf'] = conf
    db_calib['accu'] = accu
    db_calib['bsizes'] = bsizes
    report['db_calib'] = db_calib

    db_nll = nll(db_prob, ytest)
    report['db_nll'] = db_nll
    if classes == 2:
        db_typeIerror = np.mean(db_pred[ytest==0])
        report['db_typeIerror'] = db_typeIerror
        db_typeIIerror = np.mean(1-db_pred[ytest==1])
        report['db_typeIIerror'] = db_typeIIerror


    print('db_elapsed_optim =', db_elapsed_optim)
    print('db_elapsed_pred =', db_elapsed_pred)
    print('---')
    print('db_amp =', db_amp)
    print('db_len =', db_len)
    print('db_a_eps =', a_eps)
    print('---')
    print('db_error_rate =', db_error_rate)
    if classes == 2:
        print('db_typeIerror =', db_typeIerror)
        print('db_typeIIerror =', db_typeIIerror)    
    print('db_ece =', db_ece)
    print('db_nll =', db_nll)
    print('\n')
    return report


################################################################################
### Classification utilising Laplace Approximation

def evaluate_la(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None):
    report = {}
    dim = X.shape[1]
    
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)

    Y = y.reshape(y.size, 1)
    classes = np.max(y).astype(int) + 1   


    rbf_kern = gp.kern.RBF(input_dim=dim, ARD=ARD, lengthscale=default_len)
    variance = tf.Variable(1e-4, trainable=False)
    white = gp.kern.White(input_dim=dim, variance=variance)
    kernel = rbf_kern + white
    if classes==2:
        lik = gp.likelihoods.Bernoulli()
    elif classes>2:
        lik = gp.likelihoods.MultioutputLikelihood(classes)
    
    laplace = gp.inference.latent_function_inference.Laplace()
    la_model = gp.core.GP(X, Y, kernel=kernel, likelihood=lik, inference_method=laplace)
    
    if ampl is not None:
        rbf_kern.variance.trainable = False
        rbf_kern.variance = ampl * ampl
    if leng is not None:
        rbf_kern.lengthscale.trainable = False
        if ARD:
            rbf_kern.lengthscale = np.ones(dim) * leng
        else:
            rbf_kern.lengthscale = leng
    
    print('la optim... ', end='', flush=True)
    start_time = time.time()    
    la_model.optimize()  
    la_elapsed_optim = time.time() - start_time
    print('done!')
    report['la_elapsed_optim'] = la_elapsed_optim    
    
    la_amp = np.sqrt(rbf_kern.variance)
    report['la_amp'] = la_amp
    la_len = rbf_kern.lengthscale
    report['la_len'] = la_len


    # predict
    print('la pred... ', end='', flush=True)
    start_time = time.time()
    la_fmu, la_fs2 = la_model._raw_predict(Xtest)
    la_prob = np.zeros(la_fmu.shape)
    #source = np.random.randn(1000, classes)
    for i in range(la_fmu.shape[0]):
        samples = np.random.normal(la_fmu[i,:], np.sqrt(la_fs2[i,:]))
        la_prob[i,:] = samples.mean(0)
    
    
    la_elapsed_pred = time.time() - start_time
    print('done!')
    report['la_elapsed_pred'] = la_elapsed_pred



    # also get this for reference
    if classes == 2:
        la_pred = la_prob > 0.5
        la_pred = la_pred.astype(int).flatten()
    else:
        la_pred = np.argmax(la_prob, 1)

    report['la_pred'] = la_pred
    report['la_prob'] = la_prob
    report['la_fmu'] = la_fmu
    report['la_fs2'] = la_fs2


    ytest = ytest.astype(int).flatten()
    la_error_rate = np.mean(la_pred!=ytest)
    report['la_error_rate'] = la_error_rate

    la_ece, conf, accu, bsizes = calibration_test(la_prob, ytest)
    report['la_ece'] = la_ece
    la_calib = {}
    la_calib['conf'] = conf
    la_calib['accu'] = accu
    la_calib['bsizes'] = bsizes
    report['la_calib'] = la_calib

    la_nll = nll(la_prob, ytest)
    report['la_nll'] = la_nll

    if classes == 2:
        la_typeIerror = np.mean(la_pred[ytest==0])
        report['la_typeIerror'] = la_typeIerror
        la_typeIIerror = np.mean(1-la_pred[ytest==1])
        report['la_typeIIerror'] = la_typeIIerror

    print('la_elapsed_optim =', la_elapsed_optim)
    print('la_elapsed_pred =', la_elapsed_pred)
    print('---')
    print('la_amp =', la_amp)
    print('la_len =', la_len)
    print('---')
    print('la_error_rate =', la_error_rate)
    if classes == 2:
        print('la_typeIerror =', la_typeIerror)
        print('la_typeIIerror =', la_typeIIerror)
    print('la_ece =', la_ece)
    print('la_nll =', la_nll)
    print('\n')
    return report    



################################################################################
### Classification utilising Expectation propagation

def evaluate_ep(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None):
    report = {}
    dim = X.shape[1]
    
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)

    Y = y.reshape(y.size, 1)
    classes = np.max(y).astype(int) + 1   


    rbf_kern = gp.kern.RBF(input_dim=dim, ARD=ARD, lengthscale=default_len)
    variance = tf.Variable(1e-4, trainable=False)
    white = gp.kern.White(input_dim=dim, variance=variance)
    kernel = rbf_kern + white
    if classes==2:
        lik = gp.likelihoods.Bernoulli()
    elif classes>2:
        lik = gp.likelihoods.MultioutputLikelihood(classes)
    
    ep = gp.inference.latent_function_inference.EP()
    ep_model = gp.core.GP(X, Y, kernel=kernel, likelihood=lik, inference_method=ep)
    
    if ampl is not None:
        rbf_kern.variance.trainable = False
        rbf_kern.variance = ampl * ampl
    if leng is not None:
        rbf_kern.lengthscale.trainable = False
        if ARD:
            rbf_kern.lengthscale = np.ones(dim) * leng
        else:
            rbf_kern.lengthscale = leng
    
    print('ep optim... ', end='', flush=True)
    start_time = time.time()    
    ep_model.optimize()  
    ep_elapsed_optim = time.time() - start_time
    print('done!')
    report['ep_elapsed_optim'] = ep_elapsed_optim    
    
    ep_amp = np.sqrt(rbf_kern.variance)
    report['ep_amp'] = ep_amp
    ep_len = rbf_kern.lengthscale
    report['ep_len'] = ep_len


    # predict
    print('ep pred... ', end='', flush=True)
    start_time = time.time()
    ep_fmu, ep_fs2 = ep_model._raw_predict(Xtest)
    ep_prob = np.zeros(ep_fmu.shape)
    #source = np.random.randn(1000, classes)
    for i in range(ep_fmu.shape[0]):
        samples = np.random.normal(ep_fmu[i,:], np.sqrt(ep_fs2[i,:]))
        ep_prob[i,:] = samples.mean(0)
    
    
    ep_elapsed_pred = time.time() - start_time
    print('done!')
    report['ep_elapsed_pred'] = ep_elapsed_pred



    # also get this for reference
    if classes == 2:
        ep_pred = ep_prob > 0.5
        ep_pred = ep_pred.astype(int).flatten()
    else:
        ep_pred = np.argmax(ep_prob, 1)

    report['ep_pred'] = ep_pred
    report['ep_prob'] = ep_prob
    report['ep_fmu'] = ep_fmu
    report['ep_fs2'] = ep_fs2


    ytest = ytest.astype(int).flatten()
    ep_error_rate = np.mean(ep_pred!=ytest)
    report['ep_error_rate'] = ep_error_rate

    ep_ece, conf, accu, bsizes = calibration_test(ep_prob, ytest)
    report['ep_ece'] = ep_ece
    ep_calib = {}
    ep_calib['conf'] = conf
    ep_calib['accu'] = accu
    ep_calib['bsizes'] = bsizes
    report['ep_calib'] = ep_calib

    ep_nll = nll(ep_prob, ytest)
    report['ep_nll'] = ep_nll

    if classes == 2:
        ep_typeIerror = np.mean(ep_pred[ytest==0])
        report['ep_typeIerror'] = ep_typeIerror
        ep_typeIIerror = np.mean(1-ep_pred[ytest==1])
        report['ep_typeIIerror'] = ep_typeIIerror

    print('ep_elapsed_optim =', ep_elapsed_optim)
    print('ep_elapsed_pred =', ep_elapsed_pred)
    print('---')
    print('ep_amp =', ep_amp)
    print('ep_len =', ep_len)
    print('---')
    print('ep_error_rate =', ep_error_rate)
    if classes == 2:
        print('ep_typeIerror =', ep_typeIerror)
        print('ep_typeIIerror =', ep_typeIIerror)
    print('ep_ece =', ep_ece)
    print('ep_nll =', ep_nll)
    print('\n')
    return report    




def evaluate_vi(X, y, Xtest, ytest, ARD=False, Z=None, ampl=None, leng=None):
    report = {}
    dim = X.shape[1]
    
    if ARD:
        default_len = np.repeat(np.mean(np.std(X,0))*np.sqrt(dim), dim)
    else:
        default_len = np.mean(np.std(X,0))*np.sqrt(dim)

    Y = y.reshape(y.size, 1)
    classes = np.max(y).astype(int) + 1   


    rbf_kern = gp.kern.RBF(input_dim=dim, ARD=ARD, lengthscale=default_len)
    variance = tf.Variable(1e-4, trainable=False)
    white = gp.kern.White(input_dim=dim, variance=variance)
    kernel = rbf_kern + white
    if classes==2:
        lik = gp.likelihoods.Bernoulli()
    elif classes>2:
        lik = gp.likelihoods.MultioutputLikelihood(classes)
        
    if Z is None:
        vgp = gp.inference.latent_function_inference.var_gauss
        vi_model = gp.core.GP(X, Y, kernel=kernel, likelihood=lik, inference_method=vgp)
    else:
        svgp = gp.inference.latent_function_inference.svgp
        vi_model = gp.core.GP(X, Y, kernel=kernel, likelihood=lik, inference_method=svgp)
    
    if ampl is not None:
        rbf_kern.variance.trainable = False
        rbf_kern.variance = ampl * ampl
    if leng is not None:
        rbf_kern.lengthscale.trainable = False
        if ARD:
            rbf_kern.lengthscale = np.ones(dim) * leng
        else:
            rbf_kern.lengthscale = leng
    
    print('vi optim... ', end='', flush=True)
    start_time = time.time()    
    vi_model.optimize()  
    vi_elapsed_optim = time.time() - start_time
    print('done!')
    report['vi_elapsed_optim'] = vi_elapsed_optim    
    
    vi_amp = np.sqrt(rbf_kern.variance)
    report['vi_amp'] = vi_amp
    vi_len = rbf_kern.lengthscale
    report['vi_len'] = vi_len


    # predict
    print('vi pred... ', end='', flush=True)
    start_time = time.time()
    vi_fmu, vi_fs2 = vi_model._raw_predict(Xtest)
    vi_prob = np.zeros(vi_fmu.shape)
    #source = np.random.randn(1000, classes)
    for i in range(vi_fmu.shape[0]):
        samples = np.random.normal(vi_fmu[i,:], np.sqrt(vi_fs2[i,:]))
        vi_prob[i,:] = samples.mean(0)
    
    
    vi_elapsed_pred = time.time() - start_time
    print('done!')
    report['vi_elapsed_pred'] = vi_elapsed_pred



    # also get this for reference
    if classes == 2:
        vi_pred = vi_prob > 0.5
        vi_pred = vi_pred.astype(int).flatten()
    else:
        vi_pred = np.argmax(vi_prob, 1)

    report['vi_pred'] = vi_pred
    report['vi_prob'] = vi_prob
    report['vi_fmu'] = vi_fmu
    report['vi_fs2'] = vi_fs2


    ytest = ytest.astype(int).flatten()
    vi_error_rate = np.mean(vi_pred!=ytest)
    report['vi_error_rate'] = vi_error_rate

    vi_ece, conf, accu, bsizes = calibration_test(vi_prob, ytest)
    report['vi_ece'] = vi_ece
    vi_calib = {}
    vi_calib['conf'] = conf
    vi_calib['accu'] = accu
    vi_calib['bsizes'] = bsizes
    report['vi_calib'] = vi_calib

    vi_nll = nll(vi_prob, ytest)
    report['vi_nll'] = vi_nll

    if classes == 2:
        vi_typeIerror = np.mean(vi_pred[ytest==0])
        report['vi_typeIerror'] = vi_typeIerror
        vi_typeIIerror = np.mean(1-vi_pred[ytest==1])
        report['vi_typeIIerror'] = vi_typeIIerror

    print('vi_elapsed_optim =', vi_elapsed_optim)
    print('vi_elapsed_pred =', vi_elapsed_pred)
    print('---')
    print('vi_amp =', vi_amp)
    print('vi_len =', vi_len)
    print('---')
    print('vi_error_rate =', vi_error_rate)
    if classes == 2:
        print('vi_typeIerror =', vi_typeIerror)
        print('vi_typeIIerror =', vi_typeIIerror)
    print('vi_ece =', vi_ece)
    print('vi_nll =', vi_nll)
    print('\n')
    return report    