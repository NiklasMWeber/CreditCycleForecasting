# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:42:09 2022

@author: nikth
"""

from train import select_hatm_cv
from tools import add_time
from train import SignatureRegressionNik
from dataGeneration import GeneratorFermanianDependentMax
from sklearn.model_selection import train_test_split


#Xtrain, Ytrain, Xval, Yval = get_train_test_data(X_type, ntrain=ntrain, nval=nval,  Y_type=Y_type,
#                                                         npoints=npoints, d=d, scale_X=scale_X)

dimPath = 5
nPaths = 1000
num = 101

G = GeneratorFermanianDependentMax(dimPath = dimPath,nPaths = nPaths,num = num)   
G.generatePath()
X = G.X

G.generateResponse()
Y = G.Y

X_train, X_test, Y_train, Y_test = \
            train_test_split(X,Y,test_size = 0.5)

Xtimetrain = add_time(X_train)
Xtimeval = add_time(X_test)


hatm = select_hatm_cv(Xtimetrain, Y_train, scaling = True)
            
sig_reg = SignatureRegressionNik(hatm, normalizeFeatures = True)
sig_reg.fit(Xtimetrain, Y_train)

print("val.error", sig_reg.get_loss(Xtimeval, Y_test))
print("training.error", sig_reg.get_loss(Xtimetrain, Y_train))
print("val.R", sig_reg.score(Xtimeval, Y_test))
print("training.R", sig_reg.score(Xtimetrain, Y_train))

       