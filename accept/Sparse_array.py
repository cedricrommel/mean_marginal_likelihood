# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  
# Author: B. Gregorutti
# Safety Line, 2016

import numpy,sys
from sklearn.base import BaseEstimator,TransformerMixin

def Normalize(x):
	tmpx = numpy.array(x)
	norm = numpy.sqrt(numpy.dot(tmpx.T,tmpx))
	if norm == 0:
		return 0
	else:
		return tmpx / norm

def list_to_sparse_array(X):
	n = len(X)
	N = max([len(z) for z in X])
	data_sparse = numpy.empty((n,N))
	data_sparse.fill(numpy.nan)
	for k,z in enumerate(X):
		idx = range(len(z))
		data_sparse[k,idx] = Normalize(z)
	return data_sparse


class SparseArray(BaseEstimator,TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X):
		self.n = len(X)
		self.N = max([len(z) for z in X])
		return self

	def transform(self, X):
		data_sparse = numpy.empty((len(X),self.N))
		data_sparse.fill(numpy.nan)
		for k,z in enumerate(X):
			idx = range(len(z))
			data_sparse[k,idx] = Normalize(z)
		if len(X) == 1:
			return data_sparse[0]
		else:
			return data_sparse

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
