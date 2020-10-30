# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:55:55 2018

@author: Cedric Rommel - Safety Line, 2018

#################################################
FASTKDE FUNCTIONAL MARGINAL LIKELIHOODS ESTIMATOR
#################################################

This module allows to compute the vector of marginal likelihoods of a simulated
data, based on a set of real data. It is based on a nonparametric
density estimator called the Self-consistent kernel estimator. This module
features include :
* plotting the marginal likelihoods for a list of curves,
* plotting the corresponding heatmap, 
* generating animations of the pdf evolutions
* and estimating the Mean Marginal Likelihood acceptance criterion based on
the normalized pdf or on the confidence level.

Dependencies
------------
    * numpy, pandas
    * matplotlib, seaborn
    * fastkde : https://bitbucket.org/lbl-cascade/fastkde
"""

import numpy as np
import pandas as pd
#import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import sys
import os
from time import time
import seaborn as sns
from fastkde.fastKDE import fastKDE
from scipy.integrate import trapz
from scipy.interpolate import interp2d

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['font.size'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['savefig.pad_inches'] = 0.1

new_labels = [r'variable2 $V$', r'variable3 $\variable3$', r'variable4 $\variable4$',
              r'variable5 $variable5$']
new_units = [r'', r'', r'', r'']


def cut_alt(df, df_h, h1, h2):
    """ Selects data from dataframe corresponding to variable1s in given range.
    """
    if isinstance(df, np.ndarray):
        return df[(df_h >= h1)*(df_h < h2)]
    elif isinstance(df, pd.DataFrame):
        return df.loc[(df_h >= h1)*(df_h < h2), :]
    else:
        raise ValueError('ERROR: df not numpy array nor pandas DataFrame !')
        return df


def set_mask(labels_arr, labels_to_exclude):
    """ Returns boolean mask where True values correspond to labels_arr elements
    which are not contained inside labels_to_exclude array.
    """
    mask_list = [labels_arr != li for li in labels_to_exclude]
    mask = mask_list[0]
    for m in mask_list:
        mask *= m
    return mask


def vprint(msg, verbose):
    if verbose:
        print(msg)


def compute_pdf_scoring(data, h, FNUM, n_low_res_steps, n_high_res_steps,
                        n_low_res_steps2, test_datas, test_h_list=None,
                        numbDim=1, ecfPrecision=1, fracContiguousHyperVolumes=1,
                        doSaveTransformedKernel=True, normalize_pdfs=True,
                        verbose=False):
    """ Trains the and evaluates the (normalized) pdf of each test data
    conditioned on the variable1.

    Parameters
    ----------
    data : pandas DataFrame, shape = (n_samples, 4)
        Dataframe containing observations of V, variable3, variable4 and variable5 for all
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
        estimate the joint distributions of (V, variable3) and of (variable4, variable5).
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

    # Build variable1 windows depending on training data
    low_res_width1 = (3000 - h.min())/n_low_res_steps
    high_res_width = (1000)/n_high_res_steps
    low_res_width2 = (h.max() - 4000)/n_low_res_steps2
    h_arr1 = np.arange(h.min(), 3000, low_res_width1)
    h_arr2 = np.arange(3000, 4000, high_res_width)
    h_arr3 = np.arange(4000, h.max() + low_res_width2, low_res_width2)
    h_arr3[-1] += 1
    h_arr = np.hstack((h_arr1, h_arr2, h_arr3))
    h1_arr = h_arr[:-1]
    h2_arr = h_arr[1:]
    h_windows = [(h1, h2) for (h1, h2) in zip(h1_arr, h2_arr)]

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

    # Loop across variable1 windows
    pdfobj_per_h = []
    training_time_per_h = []
    pdf_per_test_data_per_h = []
    advised_max_h = h_windows[-1][-1]
    bad_bins_count = 0
    n_training_points = train.shape[0]
    for window in h_windows:

        # Select training data depending on variable1 window
        h1, h2 = window
        vprint("variable1 window: [%i, %i]\n" % (h1, h2), verbose)
        cut_train = cut_alt(train, h_train, h1, h2)
        n_obs = cut_train.shape[0]
        vprint("Number of observations: %i\n" % n_obs, verbose)
        if n_obs < 100:
            bad_bins_count += 1
            if n_obs < 15 and h2 < advised_max_h:
                advised_max_h = h2

        # Build variables iterator depending on settings
        if numbDim == 1:
            train_iterator = [cut_train.loc[:, col].values for col in train]
        elif numbDim == 2:
            train_iterator = [
                cut_train.iloc[:, [0+2*i, 1+2*i]].values.T for i in range(2)]
        elif numbDim == 4:
            train_iterator = [cut_train.values.T]
        else:
            raise ValueError(
                'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

        pdfs_per_dim = []
        training_times_per_dim = []
        # Iterate across groups of dimensions
        vprint("Training...\n", verbose)
        for train_i in train_iterator:
            start = time()
            # Instantiate fastKDE object, evaluated on regular grid, and store it
            _pdfobj = fastKDE(train_i, doSaveMarginals=False, doFFT=True,
                              positiveShift=True, logAxes=False,
                              ecfPrecision=ecfPrecision,
                              fracContiguousHyperVolumes=fracContiguousHyperVolumes,
                              doSaveTransformedKernel=doSaveTransformedKernel)
            t_time = time() - start
            pdfs_per_dim.append(_pdfobj)
            training_times_per_dim.append(t_time)
        pdfobj_per_h.append(pdfs_per_dim)
        training_time_per_h.append(np.array(training_times_per_dim))

        vprint("Predicting...\n", verbose)
        # Loop across test datas
        pdf_at_h_per_test_data_per_dim = []
        for h_test, test in zip(h_test_list, test_list):

            # Select testing data depending on variable1 window
            cut_test = cut_alt(test, h_test, h1, h2)
            if isinstance(cut_test, pd.DataFrame):
                cut_test = cut_test.values

            # Build variables iterator depending on settings
            if numbDim == 1:
                test_iterator = [cut_test[:, i] for i in range(4)]
            elif numbDim == 2:
                test_iterator = [
                    cut_test[:, [0+2*i, 1+2*i]].T for i in range(2)]
            elif numbDim == 4:
                test_iterator = [cut_test.T]
            else:
                raise ValueError(
                    'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

            # Iterate
            test_data_pdf_per_dim_at_h = []
            for test_i, _pdfobj in zip(test_iterator, pdfs_per_dim):
                # Format the test points as a list of vectors/scalars corresponding to multidimensional points
                #                if len(test_i.shape) > 1:
                #                    test_points = [test_i[i, :] for i in range(test_i.shape[0])]
                #                else:
                #                    test_points = list(test_i)

                if len(test_i.shape) == 1:
                    test_i = np.array(test_i[np.newaxis, :], dtype=np.float)
                    #test_i = test_i.reshape((test_i.size, 1))

                # Evaluate pdf and store results
#                pdf_at_points = _pdfobj.__transformphiSC_points__(test_points)
                pdf_at_points = _pdfobj.__transformphiSC_points__(test_i)
                if normalize_pdfs:
                    pdf_at_points /= np.max(_pdfobj.pdf)
                test_data_pdf_per_dim_at_h.append(pdf_at_points)

            # Get out of the loops succesively and aggregate pdfs
            pdf_at_h_per_test_data_per_dim.append(
                np.vstack((test_data_pdf_per_dim_at_h)).T)
        pdf_per_test_data_per_h.append(pdf_at_h_per_test_data_per_dim)

    # Rearrange pdfs to concatenate the pdfs for each data
    # Loop across test datas
    pdf_per_test_data = []
    scores_per_dim_list = []
    mean_scores = []
    for j in range(len(test_list)):
        # Loop across variable1s
        pdf_per_h = []
        for list_of_pdfs in pdf_per_test_data_per_h:
            # pdf array of j-th data for each variable1
            pdf_per_h.append(list_of_pdfs[j])
        pdf_for_data = np.vstack((pdf_per_h))  # whole pdf array of j-th data
        pdf_per_test_data.append(pdf_for_data)

        # Compute scores at the end, with the final pdf array(s)
        scores_per_dim = np.mean(pdf_for_data, axis=0)
        scores_per_dim_list.append(scores_per_dim)
        mean_scores.append(np.mean(scores_per_dim))

    # Calculate the variables bounds:
    var_min = train.min()
    var_max = train.max()
    var_bounds = [(m, M) for m, M in zip(var_min, var_max)]

    # Feed back to user concerning bins size and maximum variable1
    if advised_max_h < h_windows[-1][-1]:
        print("It seems that very few datas attain the higher variable1s screened. Estimations for h>%i may not be very reliable." % advised_max_h)
    if bad_bins_count > int(0.1*(n_low_res_steps+n_high_res_steps+n_low_res_steps2)):
        print("Many bins have less than 100 observations. The number of bins seem to have been set too high.")

    # Sum of the training times
    total_training_time = np.sum(np.vstack((training_time_per_h)))

    return pdf_per_test_data, scores_per_dim_list, mean_scores, pdfobj_per_h, h_train, train, h_test_list, test_list, h_windows, var_bounds, total_training_time, n_training_points


def calc_confidence_2D(x_plot, density, fx):
    mask = density < fx
    f_int = density.copy()
    f_int[mask] = 0
    x_int = x_plot[0]
    P = trapz(f_int, x_int)
    for x_int in x_plot[1:]:
        P = trapz(P, x_int)
    conf = 1 - P
    if conf < 0:
        conf = 0
    return conf


# def calc_confidence(x_plot, density, x, fx):
#    if len(x_plot)==1:
#        x_plot = x_plot[0]
#    else:
#        return calc_confidence_2D(x_plot, density, fx)
#    mask = density >= fx
#    f_int = density[mask]
#    x_int = x_plot[mask]
#    pas = x_plot[1] - x_plot[0]
#    coupures = np.where(x_int[1:] - x_int[:-1] > pas*1.1)[0]
#    if len(coupures) > 0:
#        coupure = coupures[0] + 1
#    else:
#        coupure = -1
#    P = trapz(f_int[:coupure], x_int[:coupure])
#    if len(coupures) > 1:
#        for coupure2 in coupures[1:]:
#            P += trapz(f_int[coupure:coupure2], x_int[coupure:coupure2])
#            coupure = coupure2
#    P += trapz(f_int[coupure:], x_int[coupure:])
#    return 1 - P


def compute_conf_scoring(data, h, FNUM, n_low_res_steps, n_high_res_steps,
                         n_low_res_steps2, test_datas, test_h_list=None,
                         numbDim=1, ecfPrecision=1, fracContiguousHyperVolumes=1,
                         doSaveTransformedKernel=True, normalize_pdfs=True,
                         verbose=False):
    """ Estimates the pdf of each test data conditioned on the variable1 and
    computes the confidence index of them.

    Parameters
    ----------
    data : pandas DataFrame, shape = (n_samples, 4)
        Dataframe containing observations of V, variable3, variable4 and variable5 for all
        datas. Depending on the value of test_datas, may be used only as
        training set or be split into training and testing set.

    h : numpy.ndarray, shape = (n_samples, )
        Array containing variable1 of each point in data.

    FNUM : numpy.ndarray, shape = (n_samples, )
        Array containing the data number of each data point from data.

    n_low_res_steps : integer
        Number of windows to use for cutting the variable1 up to 3000 m.

    n_high_res_steps : integer
        Number of windows to use for cutting the variable1 between 3000 and
        4000 m.

    n_low_res_steps2 : integer
        Number of windows to use for cutting the variable1 starting at 4000 m.

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
        estimate the joint distributions of (V, variable3) and of (variable4, variable5).
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

    # Build variable1 windows depending on training data
    low_res_width1 = (3000 - h.min())/n_low_res_steps
    high_res_width = (1000)/n_high_res_steps
    low_res_width2 = (h.max() - 4000)/n_low_res_steps2
    h_arr1 = np.arange(h.min(), 3000, low_res_width1)
    h_arr2 = np.arange(3000, 4000, high_res_width)
    h_arr3 = np.arange(4000, h.max() + low_res_width2, low_res_width2)
    h_arr3[-1] += 1
    h_arr = np.hstack((h_arr1, h_arr2, h_arr3))
    h1_arr = h_arr[:-1]
    h2_arr = h_arr[1:]
    h_windows = [(h1, h2) for (h1, h2) in zip(h1_arr, h2_arr)]

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

    # Loop across variable1 windows
    pdfobj_per_h = []
    conf_per_test_data_per_h = []
    training_time_per_h = []
    n_training_points = train.shape[0]
    for window in h_windows:

        # Select training data depending on variable1 window
        h1, h2 = window
        vprint("variable1 window: [%i, %i]\n" % (h1, h2), verbose)
        cut_train = cut_alt(train, h_train, h1, h2)

        # Build variables iterator depending on settings
        if numbDim == 1:
            train_iterator = [cut_train.loc[:, col].values for col in train]
        elif numbDim == 2:
            train_iterator = [
                cut_train.iloc[:, [0+2*i, 1+2*i]].values.T for i in range(2)]
        elif numbDim == 4:
            train_iterator = [cut_train.values.T]
        else:
            raise ValueError(
                'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

        pdfs_per_dim = []
        training_times_per_dim = []
        # Iterate across groups of dimensions
        vprint("Training...\n", verbose)
        for train_i in train_iterator:
            start = time()
            # Instantiate fastKDE object, evaluated on regular grid, and store it
            _pdfobj = fastKDE(train_i, doSaveMarginals=False, doFFT=True,
                              positiveShift=True, logAxes=False,
                              ecfPrecision=ecfPrecision,
                              fracContiguousHyperVolumes=fracContiguousHyperVolumes,
                              doSaveTransformedKernel=doSaveTransformedKernel)
            t_time = time() - start

            pdfs_per_dim.append(_pdfobj)
            training_times_per_dim.append(t_time)
        pdfobj_per_h.append(pdfs_per_dim)
        training_time_per_h.append(np.array(training_times_per_dim))

        vprint("Predicting...\n", verbose)
        # Loop across test datas
        conf_at_h_per_test_data_per_dim = []
        for h_test, test in zip(h_test_list, test_list):

            # Select testing data depending on variable1 window
            cut_test = cut_alt(test, h_test, h1, h2)
            if isinstance(cut_test, pd.DataFrame):
                cut_test = cut_test.values

            # Build variables iterator depending on settings
            if numbDim == 1:
                test_iterator = [cut_test[:, i] for i in range(4)]
            elif numbDim == 2:
                test_iterator = [
                    cut_test[:, [0+2*i, 1+2*i]].T for i in range(2)]
            elif numbDim == 4:
                test_iterator = [cut_test.T]
            else:
                raise ValueError(
                    'ERROR: Only 1, 2 and 4 are accepted values for argument numbDim !')

            # Iterate
            test_data_conf_per_dim_at_h = []
            for test_i, _pdfobj in zip(test_iterator, pdfs_per_dim):
                # Format the test points as a list of vectors/scalars corresponding to multidimensional points
                #                if len(test_i.shape) > 1:
                #                    test_points = [test_i[i, :] for i in range(test_i.shape[0])]
                #                else:
                #                    test_points = list(test_i)

                if len(test_i.shape) == 1:
                    test_i = np.array(test_i[np.newaxis, :], dtype=np.float)
                    #test_i = test_i.reshape((test_i.size, 1))

                # Evaluate pdf and store results
#                pdf_at_points = _pdfobj.__transformphiSC_points__(test_points)
                pdf_at_points = _pdfobj.__transformphiSC_points__(test_i)

                # For each point, compute the confidence
                confidence = np.array([
                    calc_confidence_2D(_pdfobj.axes, _pdfobj.pdf, f_y)
                    for (y, f_y) in zip(test_i.T, pdf_at_points)])

                # Set to 0 the eventual negative confidences
                #confidence[confidence<0] = 0

                test_data_conf_per_dim_at_h.append(confidence)

            # Get out of the loops succesively and aggregate pdfs
            conf_at_h_per_test_data_per_dim.append(
                np.vstack((test_data_conf_per_dim_at_h)).T)
        conf_per_test_data_per_h.append(conf_at_h_per_test_data_per_dim)

    # Rearrange pdfs to concatenate the pdfs for each data
    # Loop across test datas
    conf_per_test_data = []
    scores_per_dim_list = []
    mean_scores = []
    for j in range(len(test_list)):
        # Loop across variable1s
        conf_per_h = []
        for list_of_conf in conf_per_test_data_per_h:
            # pdf array of j-th data for each variable1
            conf_per_h.append(list_of_conf[j])
        conf_for_data = np.vstack((conf_per_h))  # whole pdf array of j-th data
        conf_per_test_data.append(conf_for_data)

        # Compute scores at the end, with the final pdf array(s)
        scores_per_dim = np.mean(conf_for_data, axis=0)
        scores_per_dim_list.append(scores_per_dim)
        mean_scores.append(np.mean(scores_per_dim))

    # Calculate the variables bounds:
    var_min = train.min()
    var_max = train.max()
    var_bounds = [(m, M) for m, M in zip(var_min, var_max)]

    # Sum of the training times
    total_training_time = np.sum(np.vstack((training_time_per_h)))

    return conf_per_test_data, scores_per_dim_list, mean_scores, pdfobj_per_h, h_train, train, h_test_list, test_list, h_windows, var_bounds, total_training_time, n_training_points


def plot_pdf_at_points(fig_name, pdf_per_test_data, data, h, FNUM,
                       test_datas, test_h_list=None, mksz=7, only_train=False, fmt='png'):
    """ Plots the results of SC density estimation evaluation at some datas.
    """

    # Split dataset into train and test set depending on arguments
    datas = list(pd.Categorical(FNUM).categories.astype(str))
    if all(isinstance(data, str) for data in test_datas):
        h_train_list = [h[FNUM == data]
                        for data in datas if data not in test_datas]
        train_list = [data.loc[FNUM == data, :]
                      for data in datas if data not in test_datas]
        h_test_list = [h[FNUM == test_name] for test_name in test_datas]
        test_list = [data.loc[FNUM == test_name, :]
                     for test_name in test_datas]
    elif isinstance(test_datas, list) and \
        (all(isinstance(data, np.ndarray) for data in test_datas) or
         all(isinstance(data, pd.DataFrame) for data in test_datas)):
        if test_h_list is not None:
            h_train_list = [h[FNUM == data] for data in datas]
            train_list = [data.loc[FNUM == data, :] for data in datas]
            h_test_list = test_h_list
            if all([test_data.shape[1] == train_list[0].shape[1] for test_data in test_datas]):
                test_list = test_datas
            else:
                raise ValueError(
                    'ERROR: Array type test_datas with incompatible shape has been encountered.')
        else:
            raise ValueError(
                'ERROR: Array type test_datas passed, but no test_h_list given.')
    else:
        raise ValueError('ERROR: unknown test_datas type passed.')

    # Set up figure
    my_dpi = 109
    #fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
    fig = plt.figure(figsize=(768/my_dpi, 1366/my_dpi), dpi=my_dpi)
    #V_ax = plt.subplot(221)
    V_ax = plt.subplot(411)
    V_ax.set_xlabel(r'variable1 $h$ $(m)$')
    V_ax.set_ylabel(r'TRUE AIRvariable2 $V$ $(m/s)$')
#    variable3_ax = plt.subplot(222)
    variable3_ax = plt.subplot(412)
    variable3_ax.set_xlabel(r'variable1 $h$ $(m)$')
    variable3_ax.set_ylabel(r'variable3 $\variable3$ $(rad)$')
#    variable4_ax = plt.subplot(223)
    variable4_ax = plt.subplot(413)
    variable4_ax.set_xlabel(r'variable1 $h$ $(m)$')
    variable4_ax.set_ylabel(r'ANGLE OF ATTACK $\variable4$ $(rad)$')
#    variable5_ax = plt.subplot(224)
    variable5_ax = plt.subplot(414)
    variable5_ax.set_xlabel('variable1 $h$ $(m)$')
    variable5_ax.set_ylabel('$variable5$ $(adim)$')

    # Loop across training datas and plot them in black ; get out of the loop
    for h_plot, train in zip(h_train_list, train_list):
        plot_V = train['V']
        plot_variable3 = train['variable3']
        plot_variable4 = train['variable4']
        plot_variable5 = train['variable5']
        V_ax.plot(h_plot, plot_V, color='black', variable4=0.1, zorder=1)
        variable3_ax.plot(h_plot, plot_variable3,
                          color='black', variable4=0.1, zorder=1)
        variable4_ax.plot(h_plot, plot_variable4,
                          color='black', variable4=0.1, zorder=1)
        variable5_ax.plot(h_plot, plot_variable5,
                          color='black', variable4=0.1, zorder=1)

    # Loop across test datas and pdfs and plots them ; get out of the loop
    if not only_train:
        for h_test, test, pdf in zip(h_test_list, test_list, pdf_per_test_data):
            test_V = test['V']
            test_variable3 = test['variable3']
            test_variable4 = test['variable4']
            test_variable5 = test['variable5']
            if test_V.shape[0] > pdf.shape[0]:
                print(
                    "Warning: PDF of lower dimension encountered. The last point of the data won't be plotted.")
                print("Test states shape: "+str(test_V.shape))
                print("Test controls shape: "+str(test_variable4.shape))
                print("Test PDF shape: "+str(pdf.shape))
                test_V = test_V[:-1]
                test_variable3 = test_variable3[:-1]
                test_variable4 = test_variable4[:-1]
                test_variable5 = test_variable5[:-1]
                h_test = h_test[:-1]

            # Distinguish between settings 1D, 2D and 4D to use good pdf columns
            if pdf.shape[1] == 4:
                plot_states = V_ax.scatter(
                    h_test, test_V, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                variable3_ax.scatter(h_test, test_variable3, s=mksz,
                                     c=pdf[:, 1], cmap='viridis', edgecolor='none', zorder=2)
                variable4_ax.scatter(h_test, test_variable4, s=mksz,
                                     c=pdf[:, 2], cmap='viridis', edgecolor='none', zorder=2)
                variable5_ax.scatter(h_test, test_variable5, s=mksz,
                                     c=pdf[:, 3], cmap='viridis', edgecolor='none', zorder=2)
            elif pdf.shape[1] == 2:
                plot_states = V_ax.scatter(
                    h_test, test_V, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                variable3_ax.scatter(h_test, test_variable3, s=mksz,
                                     c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                variable4_ax.scatter(h_test, test_variable4, s=mksz,
                                     c=pdf[:, 1], cmap='viridis', edgecolor='none', zorder=2)
                variable5_ax.scatter(h_test, test_variable5, s=mksz,
                                     c=pdf[:, 1], cmap='viridis', edgecolor='none', zorder=2)
            else:
                if len(pdf.shape) == 2:
                    if pdf.shape[1] == 1:
                        plot_states = V_ax.scatter(
                            h_test, test_V, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                        variable3_ax.scatter(
                            h_test, test_variable3, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                        variable4_ax.scatter(
                            h_test, test_variable4, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                        variable5_ax.scatter(
                            h_test, test_variable5, s=mksz, c=pdf[:, 0], cmap='viridis', edgecolor='none', zorder=2)
                    else:
                        raise ValueError(
                            'ERROR: inconsistent pdf array shape !!')
                elif len(pdf.shape) == 1:
                    plot_states = V_ax.scatter(
                        h_test, test_V, s=mksz, c=pdf, cmap='viridis', edgecolor='none', zorder=2)
                    variable3_ax.scatter(
                        h_test, test_variable3, s=mksz, c=pdf, cmap='viridis', edgecolor='none', zorder=2)
                    variable4_ax.scatter(
                        h_test, test_variable4, s=mksz, c=pdf, cmap='viridis', edgecolor='none', zorder=2)
                    variable5_ax.scatter(
                        h_test, test_variable5, s=mksz, c=pdf, cmap='viridis', edgecolor='none', zorder=2)
                else:
                    raise ValueError('ERROR: inconsistent pdf array shape !!')

    # Tidy figure and save it
    if not only_train:
        fig.tight_layout(rect=[0, 0, 0.8, 1])
        cbar_states = fig.add_axes([0.85, 0.1, 0.05, 0.8])
        fig.colorbar(plot_states, cax=cbar_states,
                     ticks=[0, 0.25, 0.5, 0.75, 1])
    else:
        fig.tight_layout()
    plt.savefig('%s.%s' % (fig_name, fmt), format=fmt)


def add_subplot_axes(ax, rect, axisbg='w'):
    """ Adds embeded subplot in axe object

    https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_density_evol(pdfobj_per_h, h, n_low_res_steps, n_high_res_steps,
                      n_low_res_steps2, var_bounds, normalize_pdfs=True, plot_kernel=True,
                      verbose=False):
    # Set variable1 sequence
    low_res_width1 = (3000 - h.min())/n_low_res_steps
    high_res_width = (1000)/n_high_res_steps
    low_res_width2 = (h.max() - 4000)/n_low_res_steps2
    h_arr1 = np.arange(h.min(), 3000, low_res_width1)
    h_arr2 = np.arange(3000, 4000, high_res_width)
    h_arr3 = np.arange(4000, h.max() + low_res_width2, low_res_width2)
    h_arr3[-1] += 1
    h_arr = np.hstack((h_arr1, h_arr2, h_arr3))
    h1_arr = h_arr[:-1]
    h2_arr = h_arr[1:]
    h_windows = [(h1, h2) for (h1, h2) in zip(h1_arr, h2_arr)]

    # Set figute
    my_dpi = 108
    plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)

    # Loop across variable1s
    for k, (window, pdfobj_per_dim) in enumerate(zip(h_windows, pdfobj_per_h)):
        h1, h2 = window
        vprint("variable1 window: [%i, %i]\n" % (h1, h2), verbose)
        m = len(pdfobj_per_dim)
        # Loop across dimensions
        for i, pdfobj in enumerate(pdfobj_per_dim):
            # Sets subplot for each one of them
            ax = plt.subplot(int(m/2), 2, i+1)
            pdf = pdfobj.pdf
            if normalize_pdfs:
                pdf /= pdf.max()
            axes = pdfobj.axes
            kernel = pdfobj.kSC
            if normalize_pdfs:
                kernel_amp = kernel.max() - kernel.min()
                kernel /= kernel_amp
            # Regular plot if 1 dim pdfs
            if len(axes) == 1:
                plt.plot(axes[0], pdf)
                plt.xlabel('%s %s' % (new_labels[i], new_units[i]))
                plt.xlim(var_bounds[i])
                if plot_kernel:
                    subax = add_subplot_axes(ax, [0.7, 0.7, 0.2, 0.2])
                    subax.plot(axes[0], kernel)
                    subax.set_xlim(var_bounds[i])
                    subax.set_title('Self-consistent kernel')
                folder = 'kde_evolution'
            # Contour plot if 2 dim pdfs (4 dim not supported yet)
            elif len(axes) == 2:
                v1, v2 = axes
                plt.contourf(v1, v2, pdf, 6, cmap='Greens')
                plt.contour(v1, v2, pdf, 6, colors='black')
#                plt.imshow(myPDF, extent=[v1.min(), v1.max(), v2.min(), v2.max()],
#                       origin='lower', cmap='Greens', aspect='auto', variable4=0.5, vmin=0,
#                       vmax=pdfmax)
                plt.colorbar()
                plt.xlabel('%s %s' % (new_labels[i*2], new_units[i*2]))
                plt.ylabel('%s %s' % (new_labels[i*2+1], new_units[i*2+1]))
                plt.xlim(var_bounds[i*2])
                plt.ylim(var_bounds[i*2+1])
                if plot_kernel:
                    subax = add_subplot_axes(ax, [0.7, 0.7, 0.2, 0.2])
                    subax.contourf(v1, v2, kernel, 6, cmap='Greens')
                    #subax.contour(v1, v2, kernel, 6, colors='black')
                    subax.set_xlim(var_bounds[i*2])
                    subax.set_ylim(var_bounds[i*2+1])
                    subax.set_title('Self-consistent kernel')
                folder = 'kde_evolution_2D'
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.suptitle('$h \in [%i$ ; $%i]$' % (h1, h2), fontsize=16)
        plt.savefig('./%s/kde_grid_%i.png' % (folder, k))
        plt.clf()


def plot_heatmap(fig_name, pdfobj_per_h, h_train, train, h_windows, conf=False, fmt='png'):
    # Set up figure
    my_dpi = 109
    fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
#    fig = plt.figure(figsize=(768/my_dpi, 1366/my_dpi), dpi=my_dpi)
    V_ax = plt.subplot(221)
#    V_ax = plt.subplot(411)
    V_ax.set_xlabel(r'variable1 $h$ $(m)$')
    V_ax.set_ylabel(r'TRUE AIRvariable2 $V$ $(m/s)$')
    variable3_ax = plt.subplot(222)
#    variable3_ax = plt.subplot(412)
    variable3_ax.set_xlabel(r'variable1 $h$ $(m)$')
    variable3_ax.set_ylabel(r'variable3 $\variable3$ $(rad)$')
    variable4_ax = plt.subplot(223)
#    variable4_ax = plt.subplot(413)
    variable4_ax.set_xlabel(r'variable1 $h$ $(m)$')
    variable4_ax.set_ylabel(r'ANGLE OF ATTACK $\variable4$ $(rad)$')
    variable5_ax = plt.subplot(224)
#    variable5_ax = plt.subplot(414)
    variable5_ax.set_xlabel('variable1 $h$ $(m)$')
    variable5_ax.set_ylabel('$variable5$ $(adim)$')
    subplots = [V_ax, variable3_ax, variable4_ax, variable5_ax]

#    # Loop across training datas and plot them in black ; get out of the loop
#    for h_plot, train in zip(h_train_list, train_list):
#        plot_V = train['V']
#        plot_variable3 = train['variable3']
#        plot_variable4 = train['variable4']
#        plot_variable5 = train['variable5']
#        V_ax.plot(h_plot, plot_V, color='black', variable4=0.1, zorder=1)
#        variable3_ax.plot(h_plot, plot_variable3, color='black', variable4=0.1, zorder=1)
#        variable4_ax.plot(h_plot, plot_variable4, color='black', variable4=0.1, zorder=1)
#        variable5_ax.plot(h_plot, plot_variable5, color='black', variable4=0.1, zorder=1)

#    hm_list = []

    # Loop across groups of variables
    for i in range(4):
        # Get axes and densities
        axes_list = [np.array(pdfobj[i].axes[0]) for pdfobj in pdfobj_per_h]
        ft_list = [pdfobj[i].pdf for pdfobj in pdfobj_per_h]
        if conf:
            marginal_scores = [[calc_confidence_2D(
                [zt_arr], ft_arr, fy) for fy in ft_arr] for zt_arr, ft_arr in zip(axes_list, ft_list)]
        else:
            marginal_scores = [ft/ft.max() for ft in ft_list]
        # Interpolate grid (because the densities do not have same resolutions)
        var = train.iloc[:, i].values
        z_interp = np.linspace(var.min(), var.max(), 300)
        ft_interp = [np.interp(z_interp, axe, ft)
                     for axe, ft in zip(axes_list, marginal_scores)]
        # Form an array
        ft_arr = np.vstack((ft_interp))
        # Get center of variable1 windows
        h_hm = np.array([np.mean(h_interv) for h_interv in h_windows])
        #hm_list.append((h_hm, z_interp))
        ax = subplots[i]
        # hm = ax.contourf(h_hm, z_interp, ft_arr.T, 500)        #plt.colorbar(hm)
        hm = ax.contourf(h_hm, z_interp, ft_arr.T, levels=np.linspace(0, 1, 200), extent=[
                         h_hm[0], h_hm[-1], z_interp[0], z_interp[-1]])  # plt.colorbar(hm)
#        hm = ax.contourf(h_hm, z_interp, ft_arr.T, levels=np.linspace(ft_arr.min(), ft_arr.max(), 200), extent=[h_hm[0], h_hm[-1], z_interp[0], z_interp[-1]])        #plt.colorbar(hm)
#        mesh_h_hm, mesh_z = np.meshgrid(h_hm, z_interp)
#        h_interp = np.linspace(h_hm[0], h_hm[-1], 300)
#        ft_call = interp2d(mesh_h_hm, mesh_z, ft_arr.T.flatten())
#        mesh_h_interp, mesh_z = np.meshgrid(h_interp, z_interp)
#        ft_interp = ft_call(mesh_h_interp, mesh_z)
#        ft_interp = ft_interp.reshape((ft_arr.T.shape))
#        hm = ax.imshow(ft_arr.T, extent=[h_hm[0], h_hm[-1], z_interp[0], z_interp[-1]], origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
#        hm = ax.imshow(ft_interp, extent=[h_hm[0], h_hm[-1], z_interp[0], z_interp[-1]], origin='lower', aspect='auto', vmin=0.0, vmax=1.0)

    # Tidy figure and save it
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    cbar_states = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(hm, cax=cbar_states, ticks=[0, 0.25, 0.5, 0.75, 1])
    plt.savefig('%s.%s' % (fig_name, fmt), format=fmt)


def compare_conf_ft(pdfobj):
    pdf = pdfobj.pdf
    axes = pdfobj.axes
    pdf /= pdf.max()
    fy_plot = np.linspace(0.0, 1.0, 100)
    conf_plot = np.array([calc_confidence_2D(axes, pdf, fy) for fy in fy_plot])
    return conf_plot


def plot_conf_ft_surface(fig_name, pdfobj_per_h, h_windows, fmt='png'):
    my_dpi = 109
#    fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
    fig = plt.figure(figsize=(768/my_dpi, 1366/my_dpi), dpi=my_dpi)
#    V_ax = plt.subplot(221, projection='3d')
    V_ax = plt.subplot(411, projection='3d')
    V_ax.set_title(r'TRUE AIRvariable2 $V$ $(m/s)$')
#    variable3_ax = plt.subplot(222, projection='3d')
    variable3_ax = plt.subplot(412, projection='3d')
    variable3_ax.set_title(r'variable3 $\variable3$ $(rad)$')
#    variable4_ax = plt.subplot(223, projection='3d')
    variable4_ax = plt.subplot(413, projection='3d')
    variable4_ax.set_title(r'ANGLE OF ATTACK $\variable4$ $(rad)$')
#    variable5_ax = plt.subplot(224, projection='3d')
    variable5_ax = plt.subplot(414, projection='3d')
    variable5_ax.set_title('$variable5$ $(adim)$')
    subplots = [V_ax, variable3_ax, variable4_ax, variable5_ax]
    h_hm = np.array([np.mean(h_interv) for h_interv in h_windows])
    fy_plot = np.linspace(0.0, 1.0, 100)
    X, Y = np.meshgrid(fy_plot, h_hm)

    for i in range(4):
        confidence = np.vstack(([compare_conf_ft(pdfobj[i])
                                 for pdfobj in pdfobj_per_h]))
        ax = subplots[i]
        ax.set_xlabel(r'Density $f(y)$')
        ax.set_ylabel(r'variable1 $h$ $(m)$')
        ax.set_zlabel(r'Confidence $\mathbb{P}(f(z)<f(y))$')
        ax.plot_wireframe(X, Y, confidence, rstride=10, cstride=10)
    fig.tight_layout()
    plt.savefig('%s_surf.%s' % (fig_name, fmt), format=fmt)

    my_dpi = 109
#    fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
    fig = plt.figure(figsize=(768/my_dpi, 1366/my_dpi), dpi=my_dpi)
#    V_ax = plt.subplot(221)
    V_ax = plt.subplot(411)
    V_ax.set_title(r'TRUE AIRvariable2 $V$ $(m/s)$')
#    variable3_ax = plt.subplot(222)
    variable3_ax = plt.subplot(412)
    variable3_ax.set_title(r'variable3 $\variable3$ $(rad)$')
#    variable4_ax = plt.subplot(223)
    variable4_ax = plt.subplot(413)
    variable4_ax.set_title(r'ANGLE OF ATTACK $\variable4$ $(rad)$')
#    variable5_ax = plt.subplot(224)
    variable5_ax = plt.subplot(414)
    variable5_ax.set_title('$variable5$ $(adim)$')
    subplots = [V_ax, variable3_ax, variable4_ax, variable5_ax]
    for i in range(4):
        confidence = np.vstack(([compare_conf_ft(pdfobj[i])
                                 for pdfobj in pdfobj_per_h]))
        ax = subplots[i]
        ax.set_ylabel(r'Density $f(y)$')
        ax.set_xlabel(r'variable1 $h$ $(m)$')
        hm = ax.imshow(confidence.T, extent=[
                       h_hm[0], h_hm[-1], 0, 1], origin='lower', aspect='auto')

    # Tidy figure and save it
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    cbar_states = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(hm, cax=cbar_states)
    plt.savefig('%s_heatmap.%s' % (fig_name, fmt), format=fmt)


def main(fig_name, data, h, FNUM, n_low_res_steps, n_high_res_steps,
         n_low_res_steps2, test_datas, numbDim=1, test_h_list=None, conf=True):

    if conf:
        ret = compute_conf_scoring(data, h, FNUM, n_low_res_steps, n_high_res_steps,
                                   n_low_res_steps2, test_datas, numbDim=numbDim,
                                   test_h_list=test_h_list)
    else:
        ret = compute_pdf_scoring(data, h, FNUM, n_low_res_steps, n_high_res_steps,
                                  n_low_res_steps2, test_datas, numbDim=numbDim,
                                  test_h_list=test_h_list)

    pdf_per_test_data, scores_per_dim, mean_scores, pdfobj_per_h, h_train, train, h_test_list, test_list, var_bounds = ret

    print("\nSCORES:\n")
    for scores_per_dim_k, mean_score in zip(scores_per_dim, mean_scores):
        for i, score in enumerate(scores_per_dim_k):
            print("Dimension %i = %.2f" % (i, score))
        print("\nMean score = %.2f \n" % mean_score)

    plot_pdf_at_points(fig_name, pdf_per_test_data, data, h, FNUM,
                       test_datas, test_h_list=test_h_list)

# if __name__ == "__main__":
#    n_low_res_steps = int(sys.argv[1])
#    n_high_res_steps = int(sys.argv[2])
#    n_low_res_steps2 = int(sys.argv[3])
#    fig = str(sys.argv[4])
#    numbDim = int(sys.argv[5])
#    main(fig, var_df, variable1, FNUM, n_low_res_steps, n_high_res_steps,
#         n_low_res_steps2, tdatas, numbDim=numbDim)
