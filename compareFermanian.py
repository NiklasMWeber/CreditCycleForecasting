# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 11:19:15 2022

@author: nikth
"""

#import sys
#sys.path.insert(0,'../')
#from utils import get_ex_results, move_legend
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd

#sns.set_theme(style="whitegrid")

from get_data import get_train_test_data
#from tools import add_time
from trainF import SignatureOrderSelection, SignatureRegression, select_hatm_cv, select_nbasis_cv, BasisRegression
#from plot_tensor_heatmap import plot_tensor_heatmap

def add_time(X):
	"""Adds a dimension with time to each smaple in X

	Parameters
	----------
	X: array, shape (n, npoints, d)
		Array of paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
		linear paths, each composed of n_points.

	Returns
	-------
	Xtime: array, shape (n, npoints, d + 1)
		Same array as X but with an extra dimension at the end, corresponding to time.
	"""
	times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))
	Xtime = np.concatenate([X, times.reshape((times.shape[0], times.shape[1], 1))], axis=2)
	return Xtime

### Create Fermanian Data
npoints = 100
Xtrain, Ytrain, Xval, Yval = get_train_test_data('smooth_dependent', ntrain=100, nval=1,  Y_type='max', npoints=npoints, d=2, scale_X=False)

Xtimetrain = Xtrain #add_time(Xtrain)
#Xtimeval = add_time(Xval)

order_sel = SignatureOrderSelection(Xtimetrain.shape[2])
print(order_sel.max_k)
hatm = order_sel.get_hatm(Xtimetrain, Ytrain, Kpen_values=np.linspace(10 ** (-10), 10 ** (2), num=200), plot=False, savefig=False)
print("Hatm", hatm)