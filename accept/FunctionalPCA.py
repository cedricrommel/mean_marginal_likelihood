# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  
# Author: B. Gregorutti
# Safety Line, 2016

"""!@module Contains the FPCA class which computes the new explicative coefficients of a vector and can project any vector on the same base.
"""

import numpy as np
import scipy.interpolate as si
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA

class FPCA:
    """!@brief Functional Principal Components Analysis for sparse and non sparse data
    """
    def __init__(self, propVar, nBasisInit, verbose=False):
        """!@brief Instantiation.
            @param x: data
            @param propVar: the percentage of explained variance for choosing the principal components
            @param nBasisInit: number of Spline coefficients we want for the smoothing step
            @param verbose: should the details be printed ?
        """
        self.propVar = propVar
        self.nBasisInit = nBasisInit
        self.verbose = verbose
        self.PCs = None
        self.idxPCs = None
        self.smoothData = None
        self.splineBasis = None
        self.eigenFunc = None
        self.pca = None
        self.reconstruct = False

    def fit(self, x):
        """!@brief Reduce the dimension of the data by computing the FPCA.
            @param reconstruct: should the curves be reconstructed after dimension reduction ?
        """
        ## STEP 1: smoothing spline

        nBasis = self.nBasisInit - 1

        dims = x.shape
        N = dims[1]
        n = dims[0]
        t = np.linspace(0., 1., N)

        # Construct Splines fonctions
        norder = 4
        nrPointsToEval = N
        nknots = nBasis - norder + 2
        knots = np.linspace(0.0, 1, nknots+2)
        
        ref = x[0,:]
        xRepres = si.splrep(t, ref, k=norder, t=knots[1:(nknots-1)])
        ipl_t = np.linspace(0., 1., nrPointsToEval)
        basis = np.zeros((nrPointsToEval,nBasis+1))
        for i in range(nBasis+1):
            coeffs = np.zeros(nBasis+1+norder+1)
            coeffs[i] = 1.0
            x_list = list(xRepres)
            x_list[1] = coeffs.tolist()
            basis[:,i] = si.splev(ipl_t, x_list)

        # Normalize the basis functions
        u,s,v = np.linalg.svd(basis, False)
        basis = u
        self.splineBasis = basis

        # Project the data on the spline basis
        tmpx = x.copy()
        idx_nan = np.where(np.isnan(tmpx))
        tmpx[np.isnan(tmpx)] = 0
        C = np.dot(tmpx, basis)
        # print C

        ## STEP 2: perform PCA on the spline coefficients
        pca = PCA()
        PCtr = pca.fit_transform(C)
        eigFunc = pca.components_ # eigen functions: array of size n_components x n_features

        explVar = np.cumsum(pca.explained_variance_ratio_)
        idxPCs = np.where(explVar <= self.propVar)[0]
        self.variances = pca.explained_variance_ratio_[idxPCs]
        if len(idxPCs)<=1:
            idxPCs = np.array([0,1])
        PCs = PCtr[:,idxPCs]
        self.PCs = PCs
        self.idxPCs = idxPCs
        self.eigenFunc = eigFunc[:,idxPCs]
        self.pca = pca

        return self


    def transform(self, newx, debug=False):

        tmpx = newx.copy()
        tmpx[np.isnan(tmpx)] = 0
        C = np.dot(tmpx, self.splineBasis)
        PCs = self.pca.transform(C)[:,self.idxPCs]
        nBasis = self.nBasisInit - 1
        nPC = PCs.shape[1]
        n = PCs.shape[0]

        if debug:
            newPCs = np.concatenate((PCs, np.zeros((n,nBasis+1-nPC))), axis=1)
            C_estimate = self.pca.inverse_transform(newPCs)
            smoothData = np.dot(C_estimate, self.splineBasis.T)
            if len(smoothData) == 1:
                smoothData = smoothData[0]
            import pandas
            pandas.DataFrame({'smoothData':smoothData, 'newx':newx}).to_csv('test_fpca_N1.csv')

        return PCs


    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, newPCs):
        dims = newPCs.shape
        n = dims[0]

        # print n, self.nBasisInit, len(self.idxPCs), self.nBasisInit - 1+1-len(self.idxPCs)
        newnewPCs = np.concatenate((newPCs, np.zeros((n,self.nBasisInit - 1+1-len(self.idxPCs)))), axis=1)
        C_estimate = self.pca.inverse_transform(newnewPCs)
        smoothData = np.dot(C_estimate, self.splineBasis.T)
                
        return smoothData
