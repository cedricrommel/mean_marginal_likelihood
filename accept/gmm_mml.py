#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:47:39 2018

@author: Cedric Rommel - Safety Line, 2018

##########################################################
GAUSSIAN MIXTURE FUNCTIONAL MARGINAL LIKELIHOODS ESTIMATOR
##########################################################

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
from sklearn import mixture
import sys
import os
from time import time
import seaborn as sns
from scipy.integrate import trapz
from scipy.stats import chi2
from scipy.interpolate import SmoothBivariateSpline

import fastkde_mml as kde

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['font.size'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['savefig.pad_inches'] = 0.1


new_labels = [r'Variable2', r'Variable3', r'Variable4',
              r'Variable5']
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


# A ENLEVER!
LABELS = ['Var2', 'Var3', 'Var4', 'Var5']


def compute_pdf_scoring(data, h, FNUM, n_low_res_steps, n_high_res_steps,
                        n_low_res_steps2, test_flights, test_h_list=None,
                        numbDim=1, ecfPrecision=1, fracContiguousHyperVolumes=1,
                        doSaveTransformedKernel=True, normalize_pdfs=True,
                        verbose=False, conf=False, analytical=True,
                        add_covs=[0, 0, 0, 0]):
    """ Trains the and evaluates the (normalized) pdf of each test flight
    conditioned on the variable1.

    Parameters
    ----------
    data : pandas DataFrame, shape = (n_samples, 4)
        Dataframe containing observations of V, variable3, variable4 and variable5 for all
        flights. Depending on the value of test_flights, may be used only as
        training set or be split into training and testing set.

    h : numpy.ndarray, shape = (n_samples, )
        Array containing variable1 of each point in data.

    FNUM : numpy.ndarray, shape = (n_samples, )
        Array containing the flight number of each data point from data.

    n_low_res_steps : integer
        Number of windows to use for cutting the variable1 up to 3000 m.

    n_high_res_steps : integer
        Number of windows to use for cutting the variable1 between 3000 and
        4000 m.

    n_low_res_steps2 : integer
        Number of windows to use for cutting the variable1 starting at 4000 m.

    test_flights : list
        List containg flight numbers (strings), numpy.ndarrays or
        pandas.DataFrames. If it contains flight number, than the corresponding
        flights will be extracted from data to form the test set, leaving the
        remaining flights for training. Otherwise, the whole dataset will be
        used for training, and testing will be performed on the points contained
        in the test_flights elements.

    test_h_list : list, optional, default=None
        If test_flights contains arrays or dataframes, then it is necessary to
        set this parameter to something else then None. It should be a list
        containing numpy.ndarrays corresponding to the variable1s of the points
        from test_flights.

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
    pdf_per_test_flight : list, len=n_test_flights
        Contains arrays storing the predicted pdfs of each test flight.

    scores_per_dim : list, len=n_test_flights
        Contains arrays of the scores for each joint distribution of each test
        flight.

    mean_score : list, len=n_test_flights
        Contains the average score over all variables for each test flight.

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
    if all(isinstance(flight, str) for flight in test_flights):
        train_mask = set_mask(FNUM, test_flights)
        h_train = h[train_mask]
        train = data.loc[train_mask, :]
        h_test_list = [h[FNUM == test_name] for test_name in test_flights]
        test_list = [data.loc[FNUM == test_name, :]
                     for test_name in test_flights]
    elif isinstance(test_flights, list) and \
        (all(isinstance(flight, np.ndarray) for flight in test_flights) or
         all(isinstance(flight, pd.DataFrame) for flight in test_flights)):
        if test_h_list is not None:
            h_train = h
            train = data
            h_test_list = test_h_list
            if all([test_flight.shape[1] == train.shape[1] for test_flight in test_flights]):
                test_list = test_flights
            else:
                raise ValueError(
                    'ERROR: Array type test_flight with incompatible shape has been encountered.')
        else:
            raise ValueError(
                'ERROR: Array type test_flights passed, but no test_h_list given.')
    else:
        raise ValueError('ERROR: unknown test_flights type passed.')

    vprint("Starting learning...\n", verbose)

    # Loop across variable1 windows
    pdfobj_per_h = []
    training_time_per_h = []
    bics_per_h = []
    pdf_per_test_flight_per_h = []
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
        bics_per_dim = []
        # Iterate across groups of dimensions
        vprint("Training...\n", verbose)
        for i, train_i in enumerate(train_iterator):
            start = time()
            # Instantiate fastKDE object, evaluated on regular grid, and store it
            lowest_bic = np.infty
            bic = []
            n_comp_max = np.min([5, train_i.shape[0]])
            n_components_range = range(1, n_comp_max)
            if len(train_i.shape) == 1:
                train_i = train_i.reshape((np.size(train_i), 1))
            else:
                train_i = train_i.T
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type='full')
                gmm.fit(train_i)
                bic.append(gmm.bic(train_i))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
            t_time = time() - start
            bic = np.array(bic)

            # JUST ADDED - DIFFUSION FEATURE
            covs = best_gmm.covariances_
            covs = [cov+add_covs[i] for cov in covs]
            best_gmm.covariances_ = covs

            pdfs_per_dim.append(best_gmm)
            training_times_per_dim.append(t_time)
            bics_per_dim.append(bic)
        pdfobj_per_h.append(pdfs_per_dim)
        bics_per_h.append(bics_per_dim)
        training_time_per_h.append(np.array(training_times_per_dim))

        vprint("Predicting...\n", verbose)
        # Loop across test flights
        pdf_at_h_per_test_flight_per_dim = []
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
            test_flight_pdf_per_dim_at_h = []
            for i, (train_i, test_i, gmm) in enumerate(zip(train_iterator, test_iterator, pdfs_per_dim)):
                # Format the test points as a list of vectors/scalars corresponding to multidimensional points
                #                if len(test_i.shape) > 1:
                #                    test_points = [test_i[i, :] for i in range(test_i.shape[0])]
                #                else:
                #                    test_points = list(test_i)

                if len(test_i.shape) == 1:
                    #test_i = np.array(test_i[np.newaxis,:],dtype=np.float)
                    test_i = test_i.reshape((np.size(test_i), 1))
                else:
                    test_i = test_i.T

                # Evaluate pdf and store results
                if test_i.shape[0] > 0:
                    if conf:
                        if analytical:
                            pdf_at_points = np.array([calc_conf_analytical(gmm, test_i[l, :])
                                                      for l in range(test_i.shape[0])])
                        else:
                            pdf_at_points = np.array([calc_conf_numerical(gmm, test_i[l, :], train_i)
                                                      for l in range(test_i.shape[0])])
                    else:
                        pdf_at_points = np.exp(gmm.score_samples(test_i))
                        if normalize_pdfs:
                            pdf_at_points /= calc_max_gmm(gmm)
                else:
                    pdf_at_points = np.array([])
                test_flight_pdf_per_dim_at_h.append(pdf_at_points)

            # Get out of the loops succesively and aggregate pdfs
            pdf_at_h_per_test_flight_per_dim.append(
                np.vstack((test_flight_pdf_per_dim_at_h)).T)
        pdf_per_test_flight_per_h.append(pdf_at_h_per_test_flight_per_dim)

    # Rearrange pdfs to concatenate the pdfs for each flight
    # Loop across test flights
    pdf_per_test_flight = []
    scores_per_dim_list = []
    mean_scores = []
    for j in range(len(test_list)):
        # Loop across variable1s
        pdf_per_h = []
        for list_of_pdfs in pdf_per_test_flight_per_h:
            # pdf array of j-th flight for each variable1
            pdf_per_h.append(list_of_pdfs[j])
        # whole pdf array of j-th flight
        pdf_for_flight = np.vstack((pdf_per_h))
        pdf_per_test_flight.append(pdf_for_flight)

        # Compute scores at the end, with the final pdf array(s)
        scores_per_dim = np.mean(pdf_for_flight, axis=0)
        scores_per_dim_list.append(scores_per_dim)
        mean_scores.append(np.mean(scores_per_dim))

    # Calculate the variables bounds:
    var_min = train.min()
    var_max = train.max()
    var_bounds = [(m, M) for m, M in zip(var_min, var_max)]

    # Sum of the training times
    total_training_time = np.sum(np.vstack((training_time_per_h)))

    return pdf_per_test_flight, scores_per_dim_list, mean_scores, pdfobj_per_h, h_train, train, h_test_list, test_list, h_windows, var_bounds, total_training_time, n_training_points


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


def calc_conf_numerical(gmm, y, train_i):
    if len(train_i.shape) > 1:
        z_grid_list = []
        for i in range(train_i.shape[0]):
            var = train_i[:, i]
            z_grid_list.append(np.linspace(var.min(), var.max(), 300))
        z_grid = np.vstack((z_grid_list)).T
        y = y.reshape((1, y.size))
        f_y = np.exp(gmm.score_samples(y))
        # density_grid doit être 2D !!
        #density_grid = np.exp(gmm.score_samples(z_grid))
        density_grid = calc_gmm_pdf(gmm, z_grid)
    else:
        z_grid = np.linspace(train_i.min(), train_i.max(), 300)
        z_grid_list = [z_grid]
        #z_grid = z_grid.reshape((z_grid.size, 1))
        y = y.reshape((y.size, 1))
        f_y = np.exp(gmm.score_samples(y)[0])
        density_grid = calc_gmm_pdf(gmm, z_grid)
        #density_grid = np.exp(gmm.score_samples(z_grid))
    return calc_confidence_2D(z_grid_list, density_grid, f_y)


def calc_conf_analytical(gmm, y):
    probas = gmm.weights_
    modes = gmm.means_
    cov_list = gmm.covariances_
    conf = []
    print("\n%i modes\n" % len(modes))
    for P, m, cov in zip(probas, modes, cov_list):
        cov_inv = np.linalg.inv(cov)
#        if y.shape[0] > m.size:
#            m = np.tile(m, y.shape[0])
#        m = m.reshape(y.shape)
#        n, p = y.shape
        p = y.size
        #region = np.dot((y - m), np.dot(cov_inv, (y - m).T))
        region = np.dot((y - m).T, np.dot(cov_inv, (y - m)))
        conf_part = chi2.cdf(region, p)
        print("\nproba: %.3f" % P)
        print("partial confidence:")
        print(conf_part)
        conf.append(P*conf_part)
    return 1 - np.sum(conf)


def plot_pdf_at_points(fig_name, pdf_per_test_flight, data, h, FNUM,
                       test_flights, test_h_list=None, mksz=7, only_train=False, fmt='png'):
    kde.plot_pdf_at_points(fig_name=fig_name, pdf_per_test_flight=pdf_per_test_flight, data=data, h=h, FNUM=FNUM,
                           test_flights=test_flights, test_h_list=test_h_list, mksz=mksz, only_train=only_train, fmt=fmt)


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


def calc_gmm_pdf(gmm, grid):
    if isinstance(grid, list):
        if len(grid) == 1:
            grid = grid[0]
            grid = grid.reshape(np.size(grid), 1)
            pdf = np.exp(gmm.score_samples(grid))
        elif len(grid) > 1:
            mesh = np.meshgrid(*grid)
            XX = np.array([X.ravel().ravel() for X in mesh]).T
            pdf = np.exp(gmm.score_samples(XX))
            pdf = pdf.reshape(mesh[0].shape)
        else:
            raise ValueError('ERROR: incompatible grid shape in gmm_pdf().')
    elif isinstance(grid, np.ndarray):
        if len(grid.shape) == 1:
            grid = grid.reshape(np.size(grid), 1)
            pdf = np.exp(gmm.score_samples(grid))
        elif len(grid.shape) == 2:
            grid = [grid[:, i] for i in range(grid.shape[1])]
            mesh = np.meshgrid(*grid)
            XX = np.array([X.ravel().ravel() for X in mesh]).T
            pdf = np.exp(gmm.score_samples(XX))
            pdf = pdf.reshape(mesh[0].shape)
        else:
            raise ValueError('ERROR: incompatible grid shape in gmm_pdf().')
    else:
        raise ValueError('ERROR: incompatible grid type in gmm_pdf().')
    return pdf


def calc_max_gmm(gmm):
    return np.max(np.exp(gmm.score_samples(gmm.means_)))


def compare_density_evol(fastkde_pdfobj_per_h, gmm_pdfobj_per_h, h, n_low_res_steps, n_high_res_steps,
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
    fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # Loop across variable1s
    for k, (window, kde_pdfobj_per_dim, gmm_pdfobj_per_dim) in enumerate(zip(h_windows, fastkde_pdfobj_per_h, gmm_pdfobj_per_h)):
        h1, h2 = window
        vprint("variable1 window: [%i, %i]\n" % (h1, h2), verbose)
        m = len(gmm_pdfobj_per_dim)
        # Loop across dimensions
        for i, (kde_pdfobj, gmm_pdfobj) in enumerate(zip(kde_pdfobj_per_dim, gmm_pdfobj_per_dim)):
            # Sets subplot for each one of them
            axes = kde_pdfobj.axes
            kde_pdf = kde_pdfobj.pdf
            gmm_pdf = calc_gmm_pdf(gmm_pdfobj, axes)

            if normalize_pdfs:
                kde_pdf /= kde_pdf.max()
                gmm_pdf /= calc_max_gmm(gmm_pdfobj)
            kernel = kde_pdfobj.kSC
            if normalize_pdfs:
                kernel_amp = kernel.max() - kernel.min()
                kernel /= kernel_amp
            # Regular plot if 1 dim pdfs
            if len(axes) == 1:
                ax = plt.subplot(int(m/2), 2, i+1)
                plt.plot(axes[0], kde_pdf)
                plt.plot(axes[0], gmm_pdf)
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
                ax1 = plt.subplot(m, 2, i+1)
                v1, v2 = axes
                plt.contourf(v1, v2, kde_pdf, 6, cmap='Greens')
                plt.contour(v1, v2, kde_pdf, 6, colors='black')
                plt.colorbar()
                plt.xlabel('%s %s' % (new_labels[i*2], new_units[i*2]))
                plt.ylabel('%s %s' % (new_labels[i*2+1], new_units[i*2+1]))
                plt.xlim(var_bounds[i*2])
                plt.ylim(var_bounds[i*2+1])
                if plot_kernel:
                    subax = add_subplot_axes(ax1, [0.7, 0.7, 0.2, 0.2])
                    subax.contourf(v1, v2, kernel, 6, cmap='Greens')
                    #subax.contour(v1, v2, kernel, 6, colors='black')
                    subax.set_xlim(var_bounds[i*2])
                    subax.set_ylim(var_bounds[i*2+1])
                    subax.set_title('Self-consistent kernel')
                ax2 = plt.subplot(m, 2, i+3)
                plt.contourf(v1, v2, gmm_pdf, 6, cmap='Blues')
                plt.contour(v1, v2, gmm_pdf, 6, colors='black')
                plt.colorbar()
                plt.xlabel('%s %s' % (new_labels[i*2], new_units[i*2]))
                plt.ylabel('%s %s' % (new_labels[i*2+1], new_units[i*2+1]))
                plt.xlim(var_bounds[i*2])
                plt.ylim(var_bounds[i*2+1])
                folder = 'kde_evolution_2D'
        plt.suptitle('$h \in [%i$ ; $%i)$' % (h1, h2), fontsize=16)
        plt.savefig('./%s/kde_grid_%i.png' % (folder, k))
        plt.clf()


def plot_heatmap(fig_name, pdfobj_per_h, h_train, train, h_windows, conf=False,
                 fmt='png'):
    # Set up figure
    my_dpi = 109
    fig = plt.figure(figsize=(1366/my_dpi, 768/my_dpi), dpi=my_dpi)
#    fig = plt.figure(figsize=(768/my_dpi, 1366/my_dpi), dpi=my_dpi)
    V_ax = plt.subplot(221)
#    V_ax = plt.subplot(411)
    V_ax.set_xlabel(r'variable1')
    V_ax.set_ylabel(r'variable2')
    variable3_ax = plt.subplot(222)
#    variable3_ax = plt.subplot(412)
    variable3_ax.set_xlabel(r'variable1')
    variable3_ax.set_ylabel(r'variable3')
    variable4_ax = plt.subplot(223)
#    variable4_ax = plt.subplot(413)
    variable4_ax.set_xlabel(r'variable1')
    variable4_ax.set_ylabel(r'variable4')
    variable5_ax = plt.subplot(224)
#    variable5_ax = plt.subplot(414)
    variable5_ax.set_xlabel('variable1')
    variable5_ax.set_ylabel('variable5')
    subplots = [V_ax, variable3_ax, variable4_ax, variable5_ax]

    # Loop across groups of variables
    for i in range(4):
        # Get axes and densities
        var = train.iloc[:, i].values
        z_interp = np.linspace(var.min(), var.max(), 300)
        ft_list = [calc_gmm_pdf(gmm_per_dim[i], z_interp)
                   for gmm_per_dim in pdfobj_per_h]
        if conf:
            marginal_scores = [[calc_conf_analytical(gmm_per_dim[i], ft_arr[l])
                                for l in range(ft_arr.shape[0])]
                               for ft_arr, gmm_per_dim in zip(ft_list, pdfobj_per_h)]
        else:
            marginal_scores = [ft_arr/calc_max_gmm(gmm_per_dim[i])
                               for ft_arr, gmm_per_dim in zip(ft_list, pdfobj_per_h)]
        # Interpolate grid (because the densities do not have same resolutions)
        # Form an array
        ft_arr = np.vstack((marginal_scores))
        # Get center of variable1 windows
        h_hm = np.array([np.mean(h_interv) for h_interv in h_windows])
        ax = subplots[i]
        hm = ax.contourf(h_hm, z_interp, ft_arr.T, levels=np.linspace(
            0, 1, 200), extent=[h_hm[0], h_hm[-1], z_interp[0], z_interp[-1]])
        # plt.colorbar(hm)

    # Tidy figure and save it
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    cbar_states = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(hm, cax=cbar_states, ticks=[0, 0.25, 0.5, 0.75, 1])
    plt.savefig('%s.%s' % (fig_name, fmt), format=fmt)


def smooth_model(pdfobj_per_h, h_windows, train, kh, kz):
    d = len(pdfobj_per_h[0])
    q = len(pdfobj_per_h)

    models = []
    for i in range(d):
        var = train.iloc[:, i].values
        h_vec = np.array([np.mean(h_windows[j]) for j in range(q)])
        z_vec = np.linspace(var.min(), var.max(), 300)
        h_grid, z_grid = np.meshgrid(h_vec, z_vec)
        h_vec2 = h_grid.T.ravel()
        z_vec2 = z_grid.T.ravel()
        f_vec = np.array([calc_gmm_pdf(gmm_per_dim[i], z_vec) /
                          calc_max_gmm(gmm_per_dim[i]) for gmm_per_dim in pdfobj_per_h])
        f_vec2 = f_vec.ravel()
        models.append(SmoothBivariateSpline(
            h_vec2, z_vec2, f_vec2, kx=kh, ky=kz))
    return models

# ev(xi, yi[, dx, dy]) 	Evaluate the spline at points
# get_coeffs() 	Return spline coefficients.
# get_knots() 	Return a tuple (tx,ty) where tx,ty contain knots positions of the spline with respect to x-, y-variable, respectively.
# get_residual() 	Return weighted sum of squared residuals of the spline
# integral(xa, xb, ya, yb) 	Evaluate the integral of the spline over area [xa,xb] x [ya,yb].


def model_to_csv(pdfobj_per_h, h_windows, direc, generic_name):
    d = len(pdfobj_per_h[0])
    q = len(pdfobj_per_h)
    if d == 4:
        labels = ['variable2', 'variable3', 'variable4', 'variable5']
    elif d == 2:
        labels = ['variable2-variable3', 'variable4-variable5']

    for i in range(d):
        outputs_i = []
        for j in range(q):
            h1, h2 = h_windows[j]
            gmm = pdfobj_per_h[j][i]
            K = int(gmm.n_components)
            output_ji = [h1, h2, K, calc_max_gmm(gmm)]
            for n in range(K):
                output_ji.append(gmm.weights_[n])
                output_ji.append(float(gmm.means_[n]))
                output_ji.append(float(gmm.covariances_[n]))
            output_ji = pd.Series(np.hstack((output_ji)))
            outputs_i.append(output_ji)
        outputs_i = pd.concat(outputs_i, axis=1)
        outputs_i.fillna(0, inplace=True)
        outputs_i.to_csv(os.path.join(direc, generic_name + '_%s.csv' % labels[i]),
                         index=False, header=False)
    h_vec = np.vstack((h_windows))[:, 0].ravel()
    h_vec = np.hstack((h_vec, h_windows[-1][-1]))
    np.savetxt(os.path.join(direc, generic_name + '_h.data'), h_vec, fmt='%i')
