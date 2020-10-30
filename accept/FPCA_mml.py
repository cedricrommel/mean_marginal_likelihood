#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:17:53 2018

@author: Cedric Rommel - Safety Line, 2018

#########################
FPCA LIKELIHOOD ESTIMATOR
#########################

This module allows to compute the acceptability of a simulated
data, based on a set of real data. It uses splines smoothing and 
Functional Principal Components Decomposition to project the data into
an interesting low and finite dimensional space, on which a density estimator
fits the points corresponding to the real data. This module features
include :


Dependencies
------------
    * numpy, pandas
    * matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from itertools import combinations
from scipy.special import comb
from scipy.interpolate import UnivariateSpline
from sklearn.mixture import GaussianMixture
from time import time

from FunctionalPCA import FPCA
from gmm_mml import calc_gmm_pdf, calc_max_gmm, vprint, set_mask


def get_datas_list(data, h, FNUM, datas):
    """ From a data array of concatenated datas, returns a list of arrays for
    each data.
    """
    data_datas_list = []
    h_datas_list = []
    for data in datas:
        mask = FNUM == data
        if isinstance(data, pd.DataFrame):
            data_datas_list.append(data.loc[mask, :].values)
        elif isinstance(data, np.array):
            data_datas_list.append(data[mask, :])
        else:
            raise ValueError(
                'Parameter data of unknown type passed to get_datas_list().')

        if isinstance(h, pd.Series):
            h_datas_list.append(h.loc[mask].values)
        elif isinstance(h, np.ndarray):
            h_datas_list.append(h[mask])
        else:
            raise ValueError(
                'Parameter h of unknown type passed to get_datas_list().')
    return data_datas_list, h_datas_list


def drop_bad_datas(data_list, h_list, datas=None, verbose=True):
    """ Finds datas whose variable1 is not strictly increasing and drops them out.
    """
    n = len(data_list)
    if datas is None:
        datas = np.arange(len(data_list))

    bad_datas = []
    datas_kept = []
    new_h_list = []
    new_data_list = []
    for i, (h, data) in enumerate(zip(h_list, data_list)):
        der_h = h[1:] - h[:-1]
        if np.any(der_h <= 0):
            bad_datas.append(datas[i])
        else:
            datas_kept.append(datas[i])
            new_data_list.append(data)
            new_h_list.append(h)
    if verbose:
        print("%i percent of datas were dropped because their variable1 was not strictly increasing." % int(
            len(bad_datas)/n*100))
        print("The new data set contains %i datas." % len(datas_kept))
    return new_data_list, new_h_list, datas_kept, bad_datas


def sparse_matrix(data_list, h_list, rough_pen, h_step=5, h_reg=None):
    """ Resamples and performs Nan-padding to the list of datas data in order
    to return a matrix of data at the same equispaced variable1s for all datas.
    """
    if h_reg is None:
        h_min = np.min([h[0] for h in h_list])
        h_max = np.max([h[-1] for h in h_list])
        h_reg = np.arange(h_min, h_max, h_step)
    new_data_list = []
    for i, (h, data) in enumerate(zip(h_list, data_list)):
        h_reg_indiv = np.array(
            [h_gre_i for h_gre_i in h_reg if (h_gre_i > h[0])*(h_gre_i < h[-1])])
        spl = UnivariateSpline(h, data, s=rough_pen)
        beg_pad = np.empty(np.sum(h_reg <= h[0]))
        beg_pad.fill(np.nan)
        end_pad = np.empty(np.sum(h_reg >= h[-1]))
        end_pad.fill(np.nan)
        new_data = np.hstack((beg_pad, spl(h_reg_indiv), end_pad))
        new_data_list.append(new_data)

    return h_reg, np.vstack((new_data_list))


def center_signals(h_reg, fun_arr, mean_signal=None, plot=False):
    """ Computes the average signal from an array containing data at the same
    equispaced variable1s.
    """
    if mean_signal is None:
        mean_signal = np.nanmean(fun_arr, axis=0)
    cent_arr = fun_arr - mean_signal

    if plot:
        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        plt.plot(h_reg, fun_arr.T, 'ko', ms=2, alpha=0.3)
        plt.plot(h_reg, mean_signal, 'r', lw=2)
        plt.subplot(212)
        plt.plot(h_reg, cent_arr.T, 'ko', ms=2, alpha=0.3)

    return cent_arr, mean_signal


def scatterplot(p_scores_train, p_scores_test=[], test_categories=[], plot_pdf=True, save_fig=False, fig_name='fpca_scatterplot.png'):
    """ Plots the scatterplot of Principal Scores for each pair of components.
    """
    plt.figure(figsize=(15, 15))
    n_comp = p_scores_train.shape[1]
    pairs = combinations(range(n_comp), 2)
    #print(int(comb(n_comp, 2)/2))
    for k, (i, j) in enumerate(pairs):
        n_comb = comb(n_comp, 2)
        n_cols = int(n_comb/2) + n_comb % 2
        plt.subplot(n_cols, 2, k+1)
        plt.scatter(p_scores_train[:, i], p_scores_train[:, j], label='train')
        if len(p_scores_test) > 0:
            if len(test_categories) == 0:
                test_categories = np.arange(len(p_scores_test))
            for p_scores, cat in zip(p_scores_test, test_categories):
                plt.scatter(p_scores[:, i], p_scores[:, j], label=cat)
            plt.legend(loc=0)
        if plot_pdf:
            grid1 = np.linspace(p_scores[:, i].min(), p_scores[:, i].max(), 50)
            grid2 = np.linspace(p_scores[:, j].min(), p_scores[:, j].max(), 50)
            data = np.vstack((p_scores[:, i], p_scores[:, j])).T
            gmm = train_gmm(data)
            gmm_pdf = calc_gmm_pdf(gmm, [grid1, grid2])/calc_max_gmm(gmm)
            plt.contour(grid1, grid2, gmm_pdf, 20, colors='black')
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.hlines(0, xmin, xmax, color='black', linestyle='-', linewidth=1)
        plt.vlines(0, ymin, ymax, color='black', linestyle='-', linewidth=1)
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.xlabel('Component %i' % (i+1))
        plt.ylabel('Component %i' % (j+1))

    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_name)


def plot_components(pca, h_reg, mean_signal, l=300, less_points=1, save_fig=False, fig_name='fpca_components.png'):
    """ Plots the principal functions and effect on the mean signal
    """
    components = np.dot(pca.splineBasis, pca.eigenFunc)
    n_comp = components.shape[1]

    plt.figure(figsize=(15, 15))
    for i in range(n_comp):
        plt.plot(
            h_reg, components[:, i], label='Component %i - %.2f' % (i+1, pca.variances[i]))
    plt.legend(loc=0)

    n_rows = int(n_comp/2) + n_comp % 2
    plt.figure(figsize=(15, 4*n_rows))
    for i, comp in enumerate(components.T):
        plt.subplot(n_rows, 2, i+1)
        plt.plot(h_reg, mean_signal)
        plt.plot(h_reg[::less_points], mean_signal[::less_points] + l *
                 components[::less_points, i], '-', label='Positive contribution')
        plt.plot(h_reg[::less_points], mean_signal[::less_points] - l *
                 components[::less_points, i], '-', label='Negative contribution')
        plt.title('Component %i - %.2f' % (i+1, pca.variances[i]))
        plt.legend(loc=0)

    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name)


def train_gmm(arr):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 5)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full')
        gmm.fit(arr)
        bic.append(gmm.bic(arr))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    return best_gmm


def compute_pdf_scoring(data, h, FNUM, all_datas, test_datas, test_h_list=None,
                        numbDim=1, normalize_pdfs=True, verbose=False, plot_fpca=False,
                        save_fig=False):

    vprint("Preparing learning...\n", verbose)

    # Split dataset into train and test set depending on arguments
    if all(isinstance(data, str) for data in test_datas):
        train_mask = set_mask(FNUM, test_datas)
        h_train = h[train_mask]
        FNUM_train = FNUM[train_mask]
        datas_train = [data for data in all_datas if data not in test_datas]
        train = data.loc[train_mask, :]
        h_test_list = [h[FNUM == test_name].values for test_name in test_datas]
        test_list = [data.loc[FNUM == test_name,
                              :].values for test_name in test_datas]
    elif isinstance(test_datas, list) and \
        (all(isinstance(data, np.ndarray) for data in test_datas) or
         all(isinstance(data, pd.DataFrame) for data in test_datas)):
        if test_h_list is not None:
            h_train = h
            train = data
            FNUM_train = FNUM
            datas_train = all_datas
            if all([test_data.shape[1] == train.shape[1] for test_data in test_datas]):
                if all(isinstance(data, pd.DataFrame) for data in test_datas):
                    test_list = [df.values for df in test_datas]
                else:
                    test_list = test_datas
                if all(isinstance(h, pd.DataFrame) for h in test_h_list):
                    h_test_list = [h.values for h in test_h_list]
                else:
                    h_test_list = test_h_list

            else:
                raise ValueError(
                    'ERROR: Array type test_data with incompatible shape has been encountered.')
        else:
            raise ValueError(
                'ERROR: Array type test_datas passed, but no test_h_list given.')
    else:
        raise ValueError('ERROR: unknown test_datas type passed.')

    start = time()

    data_datas_list, h_datas_list = get_datas_list(
        train, h_train, FNUM_train, datas_train)
    data_datas_list, h_datas_list, datas_train, bad_datas = drop_bad_datas(
        data_datas_list, h_datas_list, datas_train)

    training_size = np.sum([df.shape[0] for df in data_datas_list])

    vprint("Starting learning...\n", verbose)

    n_var = data_datas_list[0].shape[1]
    pdfs_per_dim = []
    train_pca_scores_per_dim = []
    fpca_per_dim = []
    resamp_train_per_dim = []
    mean_signal_per_dim = []
    var_labels = ['variable2', 'variable3', 'variable4', 'variable5']
    rough_pen_list = [10, 0.0001, 0.0001, 0.0001]
    l_list = [300, 0.5, 0.4, 0.5]
    propVar_list = [0.9, 0.65, 0.6, 0.75]
    # Iterate across groups of dimensions
    vprint("Training...\n", verbose)
    for i in range(n_var):
        vprint('Variable no %i: %s' % (i, var_labels[i]), verbose)
        # Selects a single variable
        train_list_i = [df[:, i] for df in data_datas_list]
        # Builds equispaced spase matrix
        h_reg, train_sparse_i = sparse_matrix(
            train_list_i, h_datas_list, rough_pen_list[i])
        # Compute mean signal and centers observations
        train_cent, mean_signal = center_signals(h_reg, train_sparse_i)
        mean_signal_per_dim.append(mean_signal)
        # Functional PCA
        fpca = FPCA(propVar_list[i], 128, verbose)
        p_scores = fpca.fit_transform(train_cent)
        if plot_fpca:
            plot_components(fpca, h_reg, mean_signal, l=l_list[i], save_fig=save_fig,
                            fig_name='fpca_components-train_%s.png' % var_labels[i])
        # Density estimation
        gmm = train_gmm(p_scores)
        # The results are stored
        resamp_train_per_dim.append(train_sparse_i)
        train_pca_scores_per_dim.append(p_scores)
        fpca_per_dim.append(fpca)
        pdfs_per_dim.append(gmm)

    total_training_time = time() - start

    vprint("Predicting...\n", verbose)

    # Drop bad test datas - XXXX CHANGE PREPROCESSING !!
    test_list, h_test_list, datas_test, bad_datas_test = drop_bad_datas(
        test_list, h_test_list)

    # Iterate accros variables
    resamp_test_per_dim = []
    test_pca_scores_per_dim = []
    test_pdf_per_dim = []
    for i, (fpca, gmm, mean_signal) in enumerate(zip(fpca_per_dim, pdfs_per_dim, mean_signal_per_dim)):
        vprint('Variable no %i: %s' % (i, var_labels[i]), verbose)
        # Selects variablesparse_matrix
        test_list_i = [df[:, i] for df in test_list]
        # Builds equispaced spase matrix
        _, test_sparse_i = sparse_matrix(
            test_list_i, h_test_list, rough_pen_list[i], h_reg=h_reg)
        # Compute mean signal and centers observations
        test_cent, _ = center_signals(
            h_reg, test_sparse_i, mean_signal=mean_signal)
        # Functional PCA
        p_scores = fpca.transform(test_cent)
        # Density eval
        pdf_at_points = np.exp(gmm.score_samples(p_scores))
        if normalize_pdfs:
            pdf_at_points /= np.max(np.exp(gmm.score_samples(gmm.means_)))
        # Store results
        test_pca_scores_per_dim.append(p_scores)
        test_pdf_per_dim.append(pdf_at_points)
        resamp_test_per_dim.append(test_sparse_i)

    # Compute mean scores across variables
    concat_scores = np.vstack((test_pdf_per_dim))
    mean_scores = concat_scores.mean(axis=0)
    concat_scores_df = pd.DataFrame(
        concat_scores, columns=datas_test, index=var_labels)
    mean_scores_df = pd.DataFrame(
        mean_scores, index=datas_test, columns=['Average']).T
    concat_scores_df = pd.concat([concat_scores_df, mean_scores_df])

    ret = {'concat_scores_df': concat_scores_df,
           'pdfs_per_dim': pdfs_per_dim,
           'fpca_per_dim': fpca_per_dim,
           'train_pca_scores_per_dim': train_pca_scores_per_dim,
           'test_pca_scores_per_dim': test_pca_scores_per_dim,
           'resamp_train_per_dim': np.vstack((resamp_train_per_dim)),
           'resamp_test_per_dim': np.vstack((resamp_test_per_dim)),
           'datas_train': datas_train,
           'bad_datas': bad_datas,
           'training_time': total_training_time,
           'training_size': training_size}

    return ret
