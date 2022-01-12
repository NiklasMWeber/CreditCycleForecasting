# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 00:03:12 2022

@author: nikth
"""

import numpy as np
import math
#from esig import tosig as ts
import iisignature as ii

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV

import dataGeneration as dg
import matplotlib.pyplot as plt



# reg = Ridge(fit_intercept=False)
# parameters = [{'alpha': [0.1, 0.25, 0.5,1, 2,5,10]}  ]
# clf = GridSearchCV(reg, parameters)
# clf.fit(sigX, Y)
# print(clf.best_params_)
# print(clf.best_score_)
#print(sorted(clf.cv_results_.keys()))

def get_sigX(X,m):
    sigX = [ii.sig(X[i],m) for i in range(len(X))]
    return np.array(sigX)

def getKpen(X,Y,max_Kpen,rho = 0.25,alpha=None,normalizeFeatures = True, plotTrue = False ):
    '''
    - Finds K_pen following Birge a d Massart,
    - alpha by Cross-validation during regression on order 1 Signature (-->
    For this reason it will be a good idea to normalize signature entries)
    - and returns the scaler to make it availbale for potential predicting.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    max_Kpen : TYPE
        DESCRIPTION.
    rho : TYPE, optional
        DESCRIPTION. The default is 0.4.
    alpha : TYPE, optional
        DESCRIPTION. The default is None.
    normalizeFeatures : TYPE, optional
        DESCRIPTION. The default is True.
    plotTrue : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    KpenVal : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    Scaler : StandardScaler
        Used to normalize data

    '''
    
    dimPath = len(X[0][0])
    nPaths = len(X)
    max_linspace = max_Kpen
    m_max = 1
    while ii.siglength(dimPath, m_max+1) < nPaths: m_max += 1
    if plotTrue == True:
        print('m_Max is '+ str(m_max))
    
    Kpen = np.linspace(1,max_linspace,max_linspace)
    KpenList = []
    losses = []
    scalers = []
    scaler = None
    
    for m in range(1,m_max+1):
        sigX = get_sigX(X,m)
        
        if normalizeFeatures == True:
            scaler = StandardScaler()
            scaler.fit(sigX)
            scalers.append(scaler)
            sigX = scaler.transform(sigX)
            
        if alpha is None: #set alpha by cross-validation in the first iteration of loop
            alphas=np.linspace(10 ** (-6), 100, num=1000)
            reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            reg_cv.fit(sigX, Y)
            alpha = reg_cv.alpha_
            
        reg = Ridge(alpha = alpha, fit_intercept=False)
        reg.fit(sigX,Y)
        predict_train = reg.predict(sigX)
        
        # reg = SignatureRegression(m,scaling = True,alpha = alpha)
        # reg.fit(np.array(X),np.array(Y))
        # predict_train = reg.predict(np.array(X))
        
        pen = Kpen.reshape((1,max_linspace))/(nPaths**rho)*math.sqrt(ii.siglength(dimPath,m))
        KpenList.append(pen)
        #squareLoss = sum((Y_test-predict_test)**2)
        squareLoss = sum((Y-predict_train)**2)/len(Y)
        losses.append(squareLoss)
        
    # The following part tries to find the first bigger jump (Birge, Massart)
    LossKpenMatrix = np.array(losses).reshape((len(losses),1))+np.array(KpenList).reshape((len(losses),max_linspace))
    mHat = np.argmin(LossKpenMatrix, axis=0)+1
    if plotTrue == True:
        plt.figure()
        plt.plot(Kpen,mHat)
    
    jumps = -mHat[1:] + mHat[:-1]
    quantile = np.quantile(jumps, 0.25)
    tmp = np.where(jumps>=max(1,quantile)) #+2 because jumps and Kpen are both legged -1 compared to value of Kpen
    try:
        KpenVal = 2*(min(tmp[0])+2)
    except:
        KpenVal = 2*max_Kpen

    return KpenVal

def getmHat(X,Y, Kpen,rho = 0.25,m_max = None,alpha=None,normalizeFeatures = True, plotTrue = False ):
    
    mHat = 1
    dimPath = len(X[0][0])
    nPaths = len(X)
    
    if m_max == None:
        m_max = 1
        while ii.siglength(dimPath, m_max+1) < nPaths: m_max += 1
        
    if plotTrue == True:
        print('m_max is '+ str(m_max))
    
    losses = []
    penalizedLosses = []
    scalers = []
    regs = []
    scaler = None
    
    for m in range(1,m_max+1):
        sigX = get_sigX(X,m)
        
        if normalizeFeatures == True:
            scaler = StandardScaler()
            scaler.fit(sigX)
            scalers.append(scaler)
            sigX = scaler.transform(sigX)
            
        if alpha is None: #select alpha by cross-validation in the first iteration of loop
            alphas=np.linspace(10 ** (-6), 100, num=1000)
            reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            reg_cv.fit(sigX, Y)
            alpha = reg_cv.alpha_
            
        reg = Ridge(alpha = alpha, fit_intercept=False)
        reg.fit(sigX,Y)
        
        
        predict_train = reg.predict(sigX)
        
        # reg = SignatureRegression(m,scaling = True,alpha = alpha)
        # reg.fit(np.array(X),np.array(Y))
        # predict_train = reg.predict(np.array(X))
        
        regs.append(reg)
        
        pen = Kpen/(nPaths**rho)*math.sqrt(ii.siglength(dimPath,m))
        #squareLoss = sum((Y_test-predict_test)**2)
        squareLoss = sum((Y-predict_train)**2)/len(Y)
        losses.append(squareLoss)
        penalizedLosses.append(squareLoss + pen)
        
    mHat = np.argmin(penalizedLosses) +1
    
    if plotTrue:
        base = np.linspace(1,m_max,num = m_max)
        plt.figure()
        plt.plot(base,penalizedLosses)

    return mHat, regs[mHat-1], scalers[mHat-1]
    

class SignatureRegression():
    """ Signature regression class

    Parameters
    ----------
    m: int
            Truncation order of the signature

    scaling: boolean, default=True
        Whether to scale the predictor matrix to have zero mean and unit variance

    alpha: float, default=None
        Regularization parameter in the Ridge regression

    Attributes
    ----------
    reg: object
        Instance of sklearn.linear_model.Ridge

    scaler: object
        Instance of sklearn.preprocessing.StandardScaler
    """

    def __init__(self, m, scaling=False, alpha=None):
        self.scaling = scaling
        self.reg = Ridge(normalize=False, fit_intercept=False, solver='svd')
        self.m = m
        self.alpha = alpha
        if self.scaling:
            self.scaler = StandardScaler()

    def fit(self, X, Y, alphas=np.linspace(10 ** (-6), 100, num=1000)):
        """Fit a signature ridge regression.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

         Y: array, shape (n)
            Array of target values.

        alphas: array, default=np.linspace(10 ** (-6), 100, num=1000)
            Grid for the cross validation search of the regularization parameter in the Ridge regression.

        Returns
        -------
        reg: object
            Instance of sklearn.linear_model.Ridge
        """

        sigX = get_sigX(X, self.m)
        if self.scaling:
            self.scaler.fit(sigX)
            sigX = self.scaler.transform(sigX)

        if self.alpha is not None:
            self.reg.alpha_ = self.alpha
        else:
            reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            reg_cv.fit(sigX, Y)
            self.alpha = reg_cv.alpha_
            self.reg.alpha_ = self.alpha
        self.reg.fit(sigX, Y)
        return self.reg

    def predict(self, X):
        """Outputs prediction of self.reg, already trained with signatures truncated at order m.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Returns
        -------
        Ypred: array, shape (n)
            Array of predicted values.
        """

        sigX = get_sigX(X, self.m)
        if self.scaling:
            sigX = self.scaler.transform(sigX)
        Ypred = self.reg.predict(sigX)
        return Ypred

    def get_loss(self, X, Y, plot=False):
        """Computes the empirical squared loss obtained with a Ridge regression on signatures truncated at m.

        Parameters
        ----------
        X: array, shape (n,n_points,d)
            Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
            linear paths, each composed of n_points.

        Y: array, shape (n)
            Array of target values.

        plot: boolean, default=False
            If True, plots the regression coefficients and a scatter plot of the target values Y against its predicted
            values Ypred to assess the quality of the fit.

        Returns
        -------
        hatL: float
            The squared loss, that is the sum of the squares of Y-Ypred, where Ypred are the fitted values of the Ridge
            regression of Y against signatures of X truncated at m.
        """
        Ypred = self.predict(X)
        if plot:
            plt.scatter(Y, Ypred)
            plt.plot([0.9 * np.min(Y), 1.1 * np.max(Y)], [0.9 * np.min(Y), 1.1 * np.max(Y)], '--', color='black')
            plt.title("Ypred against Y")
            plt.show()
        return np.mean((Y - Ypred) ** 2)
    
if __name__ == '__main__':

    dimPath = 2
    nPaths = 2000
    partition01 = np.array([j*0.01 for j in range(101)])
    mStar = 5
    
    G = dg.GeneratorFermanian1(dimPath,nPaths,mStar, num = 101)
    G.generatePath()
    G.generateResponse()
    
    #X = np.array(G.X)
    
    # add time:
    X = np.array([np.concatenate((G.partition01.reshape(-1,1), x),axis = 1) for x in G.X])
    Y = G.Y
    
    Kpen = getKpen(X,Y,max_Kpen = 2000,rho = 0.25,alpha = None,normalizeFeatures = True, plotTrue = True)
    
    mHat, reg,_ = getmHat(X, Y, Kpen, rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = True)
    
    print('Kpen: ', Kpen)
    print('m_hat: ', mHat)
    print('alpha: ', reg.alpha)
    
    