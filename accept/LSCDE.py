#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:12:41 2018

@author: cedric

https://github.com/hoxo-m/densratio_py
"""
from sklearn.base import BaseEstimator
from pylab import *
from scipy import linalg
import numpy as np

#from pyRULSIF import norm_pdf, kernel_Gaussian
#from pyRULSIF import lambda_list as compute_lambda_list
#from pyRULSIF import sigma_list as compute_sigma_list
from densratio import densratio

#class LSCDE_CV(BaseEstimator):
#    def __init__(self, lambda_list=[], kernel='gaussian', sigma_list=[], fold=5):
#        self.lambda_list = lambda_list
#        self.kernel_type = kernel
#        self.sigma_list = sigma_list
#        self.fold = fold
#
#    def fit(self, x, y, b=None, alpha=0):
#        if len(y.shape)<=1: 
#            y = y[newaxis, :]
#        if len(x.shape)<=1:
#            x = x[newaxis, :]
#        dy, ny = y.shape
#        dx, nx = x.shape
#        if dy>ny:
#            y = y.T
#            dy, ny = y.shape
#        if dx>nx:
#            x = x.T
#            dx, nx = x.shape
#
#        if nx != ny:
#            raise ValueError('Incompatible shapes for x and y !')
#    
#        if b is None:
#            b = x.shape[1]
#
#        x_nu = np.vstack((y, x))
#        x_de = x
#
#        if len(self.lambda_list) == 0:
#            self.lambda_list = compute_lambda_list()
#        if len(self.sigma_list) == 0:
#            self.sigma_list = compute_sigma_list(x_nu, x_de)
#        lambda_list = self.lambda_list
#        sigma_list = self.sigma_list
#        fold = self.fold
#
#        (d_nu,n_nu) = x_nu.shape;
#        (d_de,n_de) = x_de.shape;
#        #rand_index = permutation(n_nu);
#        b = min(b,n_nu);
#        #x_ce = x_nu[:,rand_index[0:b]]
#        self.y_min = y.min(axis=1)
#        self.y_max = y.max(axis=1)
#        #y_range = self.y_max - self.y_min
#        x_min = x.min()
#        x_max = x.max()
#        bx = int(np.sqrt(b))
#        x_ce = np.linspace(x_min, x_max, bx)
#        y_ce = np.linspace(self.y_min, self.y_max, bx)
#        X_ce, Y_ce = np.meshgrid(x_ce, y_ce)
#        x_ce_de = X_ce.ravel()[newaxis, :]
#        x_ce_nu = np.vstack((Y_ce.ravel()[newaxis, :], x_ce_de))
#    
#        score_cv=zeros( (size(sigma_list), \
#            size(lambda_list)));
#    
#        cv_index_nu = permutation(n_nu)
#        #cv_index_nu = r_[0:n_nu]
#        cv_split_nu=floor(r_[0:n_nu]*fold/n_nu)
#        cv_index_de=permutation(n_de)
#        #cv_index_de = r_[0:n_de]
#        cv_split_de=floor(r_[0:n_de]*fold/n_de)
#    
#        for sigma_index in r_[0:size(sigma_list)]:
#            sigma=sigma_list[sigma_index];
#            K_de=kernel_Gaussian(x_de,x_ce_de,sigma).T;
#            K_nu=kernel_Gaussian(x_nu,x_ce_nu,sigma).T;
#    
#            score_tmp=zeros( (fold,size(lambda_list)));
#    
#            for k in r_[0:fold]:
#                Ktmp1=K_de[:,cv_index_de[cv_split_de!=k]];
#                Ktmp2=K_nu[:,cv_index_nu[cv_split_nu!=k]];
#                
#                Ktmp = alpha/Ktmp2.shape[1]*dot(Ktmp2,Ktmp2.T) + \
#                    (1-alpha)/Ktmp1.shape[1]*dot(Ktmp1, Ktmp1.T);
#                
#                mKtmp = mean(K_nu[:,cv_index_nu[cv_split_nu!=k]],1);
#               
#                for lambda_index in r_[0:size(lambda_list)]:
#                    
#                    lbd =lambda_list[lambda_index];
#                    
#                    thetat_cv= linalg.solve( Ktmp + lbd*eye(b), mKtmp);
#                    thetah_cv=thetat_cv;
#    
#                    score_tmp[k,lambda_index]= alpha*mean(dot(K_nu[:,cv_index_nu[cv_split_nu==k]].T,thetah_cv)**2)/2. \
#                        + (1-alpha)*mean(dot(K_de[:,cv_index_de[cv_split_de==k]].T, thetah_cv)**2)/2. \
#                        - mean( dot(K_nu[:,cv_index_nu[cv_split_nu==k]].T, thetah_cv));
#    
#                score_cv[sigma_index,:]=mean(score_tmp,0);
#        
#        score_cv_tmp=score_cv.min(1);
#        lambda_chosen_index = score_cv.argmin(1);
#    
#        score=score_cv_tmp.min();
#        sigma_chosen_index = score_cv_tmp.argmin();
#    
#        lambda_chosen=lambda_list[lambda_chosen_index[sigma_chosen_index]];
#        sigma_chosen=sigma_list[sigma_chosen_index];
#    
#        K_de=kernel_Gaussian(x_de,x_ce_de,sigma_chosen).T;
#        K_nu=kernel_Gaussian(x_nu,x_ce_nu,sigma_chosen).T;
#    
#        coe = alpha*dot(K_nu,K_nu.T)/n_nu + \
#            (1-alpha)*dot(K_de,K_de.T)/n_de + \
#            lambda_chosen*eye(b)
#        var = mean(K_nu,1)
#        
#        thetat=linalg.solve(coe,var);
#        self.coefs_ = thetat
#        self.score_cv = score_cv
#        self.best_score = score
##        self.K_de_ = K_de
##        self.K_nu_ = K_nu
#        self.sigma_ = sigma_chosen
#        self.lambda_ = lambda_chosen
#        self.x_ce_de_ = x_ce_de
#        self.x_ce_nu_ = x_ce_nu
#
#
#    def predict(self, x_re, y_re):
##        K_de = self.K_de_
##        K_nu = self.K_nu_
#        thetah = self.coefs_
#        sigma_chosen = self.sigma_
##        x_ce_de = self.x_ce_de_
#        x_ce_nu = self.x_ce_nu_
#
##        wh_x_de=dot(K_de.T,thetah).T;
##        wh_x_nu=dot(K_nu.T,thetah).T;
#    
#        K_di=kernel_Gaussian(x_re, x_ce_nu, sigma_chosen).T;
#        wh_x_re=dot(K_di.T,thetah).T;
#    
##        wh_x_de[wh_x_de <0 ] = 0
#        wh_x_re[wh_x_re <0] = 0;
#    
#        # Added
##        x_re_de = x_re[0,:]
##        x_re_de = x_re_de[:, np.newaxis].T
##        K_re_de=kernel_Gaussian(x_re_de,x_ce_de,sigma_chosen).T;
##        wh_x_re_de=dot(K_re_de.T,thetah).T;
##        #normalization = (np.sqrt(2*np.pi)*sigma_chosen)*wh_x_re_de
##        normalization = wh_x_re_de
##        wh_x_re /= normalization
#        return wh_x_re
#
#
#    def max(self, x_re, y_min=None, y_max=None, n_steps=1000):
#        if y_min is None:
#            y_min = self.y_min
#        if y_max is None:
#            y_max = self.y_max
#        y_re = np.linspace(y_min, y_max, n_steps)
#        max_vec = np.array([np.max(self.predict(x, y_re)) for x in x_re])
#        return max_vec


#class LSCDE_CV_new(BaseEstimator):
class LSCDE_CV_new(BaseEstimator):
    def __init__(self, lambda_list=[], kernel='gaussian', sigma_list=[], fold=5):
        self.lambda_list = lambda_list
        self.kernel_type = kernel
        self.sigma_list = sigma_list
        self.fold = fold

    def fit(self, x, y, sigma_range=None, lambda_range=None, kernel_num=None):
#        ny, dy = y.shape
#        nx, dx = x.shape
#        if dy>ny:
#            y = y.T
#            ny, dy = y.shape
#        if dx>nx:
#            x = x.T
#            nx, dx = x.shape

        if np.any(x.shape != y.shape):
            raise ValueError('Incompatible shapes for x and y !')
    
        self.y_min = y.min()
        self.y_max = y.max()
        R = self.y_max - self.y_min
        dummy = np.random.rand(*x.shape)*2*R - R
        x_nu = np.vstack((y,x)).T
        x_de = np.vstack((dummy,x)).T
        if kernel_num is None:
            kernel_num=100
        if sigma_range is None or lambda_range is None:
            model = densratio(x_nu, x_de, kernel_num=kernel_num)
        else:
            model = densratio(x_nu, x_de, sigma_range, lambda_range, kernel_num=kernel_num)
        self.model = model
#        self.output_range = R


    def predict(self, x_eval, y_eval):
        model = self.model
#        R = self.output_range
        R = self.y_max - self.y_min

        x_re = np.vstack((y_eval, x_eval)).T

        wh_x_re = model.compute_density_ratio(x_re)/(2*R)

        wh_x_re[wh_x_re <0] = 0

        return wh_x_re


    def max(self, x_re, y_min=None, y_max=None, n_steps=300):
        if y_min is None:
            y_min = self.y_min
        if y_max is None:
            y_max = self.y_max
        y_re = np.linspace(y_min, y_max, n_steps)

        max_vec = np.array([np.max(self.predict(x*np.ones(n_steps), y_re)) for x in x_re])
        if np.any(max_vec==0):
            print('Flat density encountered (max=0)!')
            max_vec[max_vec==0] = 10**-7
        
        return max_vec
