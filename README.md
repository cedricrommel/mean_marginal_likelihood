# Mean marginal likelihood

**Important note**: This package dependends on the package _fastkde_,
which must have been installed previously. It also needs _densratio_py_
if you want to use LSCDE (otherwise, just delete it from the package
directory and init file).

The purpose of this package is to provide tools for computing trajectory
acceptability scores based on the Mean Marginal Likelihood. More
information on this criteria and its uses can be found in the following
papers: [[1]](https://hal.inria.fr/hal-01819749/document), [[2]](https://hal.inria.fr/hal-01816407/document). This package
contains classes and methods allowing to \* compute marginal densities
from a set of trajectories, \* plot the corresponding heatmap, \*
compute the overall mean marginal likelihood scores of test trajectories
using normalized densities scaling and confidence level scaling, \* plot
the marginal likelihoods point-by-point for a list of curves, ...

While the module _fastkde_mml_ computes the MML based on a
nonparametric density estimator called the Self-consistent kernel
estimator (see [[3]](https://www.jstor.org/stable/41262677?seq=1)), Gaussian Mixtures Models (whose number of components is
set between 1 and 5 using BIC) are used in the module _gmm_mml_. The
later allows to export the parameters of each GMM as a csv file.

Modules LSCDE_mml and FPCA_mml contain code capable of computing
trajectories likelihood based on Least-Squares Conditional Density
Estimation and Functional PCA (see comparison results in
this article [[1]](https://hal.inria.fr/hal-01819749/document)).

### Pre-requisites

- **Python 2.7**
- sklearn
- pandas
- seaborn
- joblib
- Numpy
- [fastkde](https://bitbucket.org/lbl-cascade/fastkde/src/master/)
- [densratio_py](https://github.com/hoxo-m/densratio_py)

## Overview

A tutorial notebook can be found in this same directory, under the name
_Tutorial_traj_accept.ipynb_.
