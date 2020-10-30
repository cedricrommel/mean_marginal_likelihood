#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:18:39 2018

@author: Cedric Rommel - Safety Line, 2018

##############################################################################
LEAST-SQAURES CONDITIONAL DESITY ESTIMATOR FOR FUNCTIONAL MARGINAL LIKELIHOODS
##############################################################################

This module allows to compute the vector of marginal likelihoods of a simulated
data, based on a set of real data. It is based on a parametric
density estimator : Gaussian Mixture model fitted using a frequentist EM
algorithm. This module features include :
* plotting the marginal likelihoods for a list of curves,
* plotting the corresponding heatmap,
* generating animations of the pdf evolutions,
* estimating the Mean Marginal Likelihood acceptance criterion based on
the normalized pdf or on the confidence level,
* the generation of a csv containing the model parameters for intefacing it with
BOCOP optimizer.

Dependencies
------------
    * numpy, pandas
    * matplotlib, seaborn
    * fastkde : https://bitbucket.org/lbl-cascade/fastkde
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from time import time
import seaborn as sns
from LSCDE import LSCDE_CV_new  # LSCDE_CV
from fastkde_mml import set_mask


def vprint(msg, verbose):
    if verbose:
        print(msg)


def compute_pdf_scoring(data, h, FNUM, test_datas, test_h_list=None,
                        numbDim=1, normalize_pdfs=True, verbose=False):
    """ Trains the and evaluates the (normalized) pdf of each test data
    conditioned on the variable1.

    Parameters
    ----------
    data : pandas DataFrame, shape = (n_samples, 4)
        Dataframe containing observations for all
        datas. Depending on the value of test_datas, may be used only as
        training set or be split into training and testing set.

    h : numpy.ndarray, shape = (n_samples, )
        Array containing variable1 of each point in data.

    FNUM : numpy.ndarray, shape = (n_samples, )
        Array containing the data number of each data point from data.

    n_low_res_steps : integer
        Number of windows to use for cutting the variable1

    n_high_res_steps : integer
        Number of windows to use for cutting the variable1

    n_low_res_steps2 : integer
        Number of windows to use for cutting the variable1

    test_datas : list
        List containg data numbers (strings), numpy.ndarrays or
        pandas.DataFrames. If it contains data number, than the corresponding
        datas will be extracted from data to form the test set, leaving the
        remaining datas for training. Otherwise, the whole dataset will be
        used for training, and testing will be performed on the points contained
        in the test_datas elements.

    test_h_list : list, optional, default=None
        If test_datas contains arrays or dataframes, then it is necessary to
        set this parameter to something else then None. It should be a list
        containing numpy.ndarrays corresponding to the variable1s of the points
        from test_datas.

    numbDim : integer, accepted values are 1, 2 or 4
        Defines how many variables to use for joint density estimation. For
        example, if set to 1, univariate density estimation is performed for
        each of the 4 variables contained in data. If set to 2, it will
        estimate the joint distributions of (var2, var3) and of (var4, var5).
        **WARNING: Memory Error is probable if set to 4 !!**

    ecfPrecision : integer
        Sets the precision of the approximate ECF.  If set to 2, it uses double
        precision accuracy; 1 otherwise.

    fracContiguousHyperVolumes : float
        The fraction of contiguous hypervolumes of the ECF, that are above the
        ECF threshold, to use in the density estimate.

    doSaveTransformedKernel : boolean
        Flags whether to keep a copy of the SC kernel in the fastKDE object.
        **Not implemented yet !!**

    normalize_pdfs : boolean
        Flags whether to normalize to 1 the estimated pdfs.

    verbose : boolean

    Returns
    -------
    pdf_per_test_data : list, len=n_test_datas
        Contains arrays storing the predicted pdfs of each test data.

    scores_per_dim : list, len=n_test_datas
        Contains arrays of the scores for each joint distribution of each test
        data.

    mean_score : list, len=n_test_datas
        Contains the average score over all variables for each test data.

    pdfobj_per_h : list
        Contains trained fastKDE objects for each variable1 window. These can be
        used to plot the pdf through the objects axes and pdf, or to plot the
        self-consistent kernels when doSaveTransformedKernel is set to True.

    h_train : array
        Contains the variable1 of the training points used.

    train : pandas.DataFrame
        Contains the training points used.

    h_test_list : list
        Contains arrays of the variable1s of the test points.

    test_list : list
        Contains array-like elements storing the test points.

    """

    vprint("Preparing learning...\n", verbose)

    # Split dataset into train and test set depending on arguments
    if all(isinstance(data, str) for data in test_datas):
        train_mask = set_mask(FNUM, test_datas)
        h_train = h[train_mask]
        train = data.loc[train_mask, :]
        h_test_list = [h[FNUM == test_name] for test_name in test_datas]
        test_list = [data.loc[FNUM == test_name, :]
                     for test_name in test_datas]
    elif isinstance(test_datas, list) and \
        (all(isinstance(data, np.ndarray) for data in test_datas) or
         all(isinstance(data, pd.DataFrame) for data in test_datas)):
        if test_h_list is not None:
            h_train = h
            train = data
            h_test_list = test_h_list
            if all([test_data.shape[1] == train.shape[1] for test_data in test_datas]):
                test_list = test_datas
            else:
                raise ValueError(
                    'ERROR: Array type test_data with incompatible shape has been encountered.')
        else:
            raise ValueError(
                'ERROR: Array type test_datas passed, but no test_h_list given.')
    else:
        raise ValueError('ERROR: unknown test_datas type passed.')

    vprint("Starting learning...\n", verbose)

    # Build variables iterator depending on settings
    if numbDim == 1:
        train_iterator = [train.loc[:, col].values for col in train]
    elif numbDim == 2:
        train_iterator = [
            train.iloc[:, [0+2*i, 1+2*i]].values.T for i in range(2)]
    elif numbDim == 4:
        train_iterator = [train.values.T]
    else:
        raise ValueError(
            'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

    pdfs_per_dim = []
    n_training_points = train.shape[0]
    # Iterate across groups of dimensions
    vprint("Training...\n", verbose)
    start = time()
    for train_i in train_iterator:
        # Instantiate fastKDE object, evaluated on regular grid, and store it
        #        hyperparam_search = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        #        model = LSCDE_CV(lambda_list = hyperparam_search, sigma_list=hyperparam_search).fit(h_train, train_i)
        #model = LSCDE_CV_new().fit(h_train, train_i)
        model = LSCDE_CV().fit(h_train, train_i)

        pdfs_per_dim.append(model)
    training_time = time() - start

    vprint("Predicting...\n", verbose)
    # Loop across test datas
    pdf_per_test_data = []
    for h_test, test in zip(h_test_list, test_list):

        # Build variables iterator depending on settings
        if numbDim == 1:
            test_iterator = [test[:, i] for i in range(4)]
        elif numbDim == 2:
            test_iterator = [test[:, [0+2*i, 1+2*i]].T for i in range(2)]
        elif numbDim == 4:
            test_iterator = [test.T]
        else:
            raise ValueError(
                'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

        # Iterate
        test_data_pdf_per_dim = []
        for i, (train_i, test_i, model) in enumerate(zip(train_iterator, test_iterator, pdfs_per_dim)):

            if len(test_i.shape) == 1:
                #test_i = np.array(test_i[np.newaxis,:],dtype=np.float)
                test_i = test_i.reshape((np.size(test_i), 1))
            else:
                test_i = test_i.T

            # Evaluate pdf and store results
            if test_i.shape[0] > 0:
                pdf_at_points = model.predict(h_test, test_i)
                if normalize_pdfs:
                    pdf_at_points /= model.max(h_test)
            else:
                pdf_at_points = np.array([])
            test_data_pdf_per_dim.append(pdf_at_points)

        # Get out of the loops succesively and aggregate pdfs
        pdf_per_test_data.append(np.vstack((test_data_pdf_per_dim)).T)

    # Rearrange pdfs to concatenate the pdfs for each data
    # Loop across test datas
    pdf_per_test_data = []
    scores_per_dim_list = []
    mean_scores = []

    for pdf_for_data in pdf_per_test_data:
        scores_per_dim = np.mean(pdf_for_data, axis=0)
        scores_per_dim_list.append(scores_per_dim)
        mean_scores.append(np.mean(scores_per_dim))

    # Calculate the variables bounds:
    var_min = train.min()
    var_max = train.max()
    var_bounds = [(m, M) for m, M in zip(var_min, var_max)]

    ret = {'pdf_per_test_data': pdf_per_test_data,
           'scores_per_dim_list': scores_per_dim_list,
           'mean_scores': mean_scores,
           'pdfs_per_dim': pdfs_per_dim,
           'h_train': h_train,
           'train': train,
           'h_test_list': h_test_list,
           'test_list': test_list,
           'var_bounds': var_bounds,
           'training_time': training_time,
           'training_size': n_training_points}

    return ret
