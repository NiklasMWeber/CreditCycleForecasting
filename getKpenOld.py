# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 00:03:12 2022

@author: nikth
"""

import numpy as np
import math
from esig import tosig as ts
import iisignature as ii

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import dataGeneration as dg
import matplotlib.pyplot as plt

dimPath = 2
nPaths = 2000
partition01 = np.array([j*0.01 for j in range(101)])
mStar = 5

G = dg.GeneratorFermanian1(dimPath,nPaths,partition01,mStar)
G.generatePath()
G.generateResponse()

X = G.X
#X = [np.concatenate((G.partition01.reshape(-1,1), x),axis = 1) for x in G.X]
Y = G.Y

# ridge_reg = Ridge(fit_intercept=False)
# parameters = [{'alpha': [0.1, 0.25, 0.5,1, 2,5,10]}  ]
# clf = GridSearchCV(ridge_reg, parameters)
# clf.fit(sigX, Y)
# print(clf.best_params_)
# print(clf.best_score_)
#print(sorted(clf.cv_results_.keys()))

plotTrue = True
testPercent = 0.2
max_linspace = 2000
alpha = 2
rho = 0.4
m_max = 2
while ii.siglength(G.dimPath, m_max+1) < G.nPaths*(1-testPercent): m_max += 1
print('m_Max is '+ str(m_max))

Kpen = np.linspace(1,max_linspace,max_linspace)
KpenList = []
losses = []

train_idx, test_idx, Y_train, Y_test = train_test_split(range(len(X)),Y, test_size = testPercent)
for m in range(1,m_max+1):
    sigX = [ii.sig(G.X[i], m) for i in range(G.nPaths)]
    X_train, X_test = [sigX[i] for i in train_idx], [sigX[i] for i in test_idx]
    ridge_reg = Ridge(alpha = alpha, fit_intercept=False)
    ridge_reg.fit(X_train,Y_train)
    
    predict_test = ridge_reg.predict(X_test)
    predict_train = ridge_reg.predict(X_train)
    
    pen = Kpen.reshape((1,max_linspace))/(G.nPaths**rho)*math.sqrt(ii.siglength(G.dimPath,m))
    KpenList.append(pen)
    #squareLoss = sum((Y_test-predict_test)**2)
    squareLoss = sum((Y_train-predict_train)**2)/len(Y_train)
    losses.append(squareLoss)
    
LossKpenMatrix = np.array(losses).reshape((len(losses),1))+np.array(KpenList).reshape((len(losses),max_linspace))
mHat = np.argmin(LossKpenMatrix, axis=0)+1
if plotTrue == True:
    plt.plot(Kpen,mHat)

jumps = -mHat[1:] + mHat[:-1]
quantile = np.quantile(jumps, 0.25)
KpenVal = 2*(min(np.where(jumps>=max(1,quantile))[0])+2) #+2 because jumps and Kpen are both legged -1 compared to value of Kpen


    
    