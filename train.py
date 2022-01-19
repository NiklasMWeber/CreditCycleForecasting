# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 00:03:12 2022

@author: nikth
"""

import numpy as np
import math
import iisignature as ii

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def get_sigX(X,m):
    if m == 0:
        return np.full((np.shape(X)[0], 1), 1)
    else:
        d = X.shape[2]
        sigX = np.zeros((np.shape(X)[0], ii.siglength(d, m) + 1))
        sigX[:, 0] = 1
        for i in range(np.shape(X)[0]):
	        sigX[i, 1:] = ii.sig(X[i, :, :], m)
        return sigX

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
    m_max = 1
    while ii.siglength(dimPath, m_max+1) < nPaths: m_max += 1
    if plotTrue == True:
        print('m_Max is '+ str(m_max))
    
    Kpen = np.concatenate(( np.array([1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1])  ,np.linspace(1,max_Kpen,max_Kpen)))
    penList = []
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
        
        pen = Kpen.reshape((1,len(Kpen)))/(nPaths**rho)*math.sqrt(ii.siglength(dimPath,m))
        penList.append(pen)
        #squareLoss = sum((Y_test-predict_test)**2)
        squareLoss = sum((Y-predict_train)**2)/len(Y)
        losses.append(squareLoss)
        
    # The following part tries to find the first bigger jump (Birge, Massart)
    LossKpenMatrix = np.array(losses).reshape((len(losses),1))+np.array(penList).reshape((len(losses),len(Kpen)))
    mHat = np.argmin(LossKpenMatrix, axis=0)+1
    if plotTrue == True:
        plt.figure()
        plt.plot(np.linspace(1,len(Kpen), num = len(Kpen)),mHat)
    
    jumps = -mHat[1:] + mHat[:-1]
    quantile = np.quantile(jumps, 0.25)
    tmp = np.where(jumps>=max(1,quantile))
    try:
        #tmp2 = tmp[0]
        KpenVal = 2*(Kpen[min(tmp[0])+1])
    except:
        KpenVal = 2*Kpen[0]
        print("Warning: No jumps for Kpen extraction found")

    return KpenVal

def getmHat(X,Y, Kpen,rho = 0.25,m_max = None,alpha=None,normalizeFeatures = True, plotTrue = False, mHatInput= None ):
    
    mHat = 1
    dimPath = len(X[0][0])
    nPaths = len(X)
    
    if m_max == None:
        m_max = 1
        while ii.siglength(dimPath, m_max+1) < nPaths*10: m_max += 1
        
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

    if mHatInput == None:
        return mHat, regs[mHat-1], scalers[mHat-1]
    else:
        mHatInput = min(mHatInput,m_max)
        return mHatInput, regs[mHatInput-1], scalers[mHatInput-1]
    
    
def select_hatm_cv(X, Y, max_k=None, scaling=False, plot=False):
    """Select the optimal value of hatm for the signature linear model implemented in the class SignatureRegression by
    cross validation.

    Parameters
    ----------
    X: array, shape (n,n_points,d)
        Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
        linear paths, each composed of n_points.

    Y: array, shape (n)
        Array of target values.

    max_k: int,
        Maximal value of signature truncation to keep the number of features below max_features.

    scaling: boolean, default=False
        Whether to scale the predictor matrix to have zero mean and unit variance

    plot: boolean, default=False
        If true, plot the cross validation loss as a function of the truncation order.

    Returns
    -------
    hatm: int
        Optimal value of hatm.
    """
    d = X.shape[2]
    max_features = 10 ** 4
    if max_k is None:
        max_k = math.floor((math.log(max_features * (d - 1) + 1) / math.log(d)) - 1)
    score = []
    
    sigXmax = get_sigX(X,max_k)
    
    for k in range(max_k+1):
        if k == 0: 
            siglength = 0 #this is length without level 0 one!
        else:
            siglength = ii.siglength(d,k)
        sigX = sigXmax[:,0:siglength+1]
        kf = KFold(n_splits=5)
        score_i = []
        for train, test in kf.split(X):
            reg = SignatureRegressionNik(k, normalizeFeatures=scaling)
            reg.fit_fromSig(sigX[train], Y[train])
            score_i += [reg.get_loss_fromSig(sigX[test], Y[test])]
        score += [np.mean(score_i)]
    if plot:
        plt.plot(np.arange(max_k+1), score)
        plt.show()
    return np.argmin(score)
    

# class SignatureRegression():
#     """ Signature regression class

#     Parameters
#     ----------
#     m: int
#             Truncation order of the signature

#     scaling: boolean, default=True
#         Whether to scale the predictor matrix to have zero mean and unit variance

#     alpha: float, default=None
#         Regularization parameter in the Ridge regression

#     Attributes
#     ----------
#     reg: object
#         Instance of sklearn.linear_model.Ridge

#     scaler: object
#         Instance of sklearn.preprocessing.StandardScaler
#     """

#     def __init__(self, m, scaling=False, alpha=None):
#         self.scaling = scaling
#         self.reg = Ridge(normalize=False, fit_intercept=False, solver='svd')
#         self.m = m
#         self.alpha = alpha
#         if self.scaling:
#             self.scaler = StandardScaler()

#     def fit(self, X, Y, alphas=np.linspace(10 ** (-6), 100, num=1000)):
#         """Fit a signature ridge regression.

#         Parameters
#         ----------
#         X: array, shape (n,n_points,d)
#             Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
#             linear paths, each composed of n_points.

#          Y: array, shape (n)
#             Array of target values.

#         alphas: array, default=np.linspace(10 ** (-6), 100, num=1000)
#             Grid for the cross validation search of the regularization parameter in the Ridge regression.

#         Returns
#         -------
#         reg: object
#             Instance of sklearn.linear_model.Ridge
#         """

#         sigX = get_sigX(X, self.m)
#         if self.scaling:
#             self.scaler.fit(sigX)
#             sigX = self.scaler.transform(sigX)

#         if self.alpha is not None:
#             self.reg.alpha_ = self.alpha
#         else:
#             reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
#             reg_cv.fit(sigX, Y)
#             self.alpha = reg_cv.alpha_
#             self.reg.alpha_ = self.alpha
#         self.reg.fit(sigX, Y)
#         return self.reg

#     def predict(self, X):
#         """Outputs prediction of self.reg, already trained with signatures truncated at order m.

#         Parameters
#         ----------
#         X: array, shape (n,n_points,d)
#             Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
#             linear paths, each composed of n_points.

#         Returns
#         -------
#         Ypred: array, shape (n)
#             Array of predicted values.
#         """

#         sigX = get_sigX(X, self.m)
#         if self.scaling:
#             sigX = self.scaler.transform(sigX)
#         Ypred = self.reg.predict(sigX)
#         return Ypred

#     def get_loss(self, X, Y, plot=False):
#         """Computes the empirical squared loss obtained with a Ridge regression on signatures truncated at m.

#         Parameters
#         ----------
#         X: array, shape (n,n_points,d)
#             Array of training paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
#             linear paths, each composed of n_points.

#         Y: array, shape (n)
#             Array of target values.

#         plot: boolean, default=False
#             If True, plots the regression coefficients and a scatter plot of the target values Y against its predicted
#             values Ypred to assess the quality of the fit.

#         Returns
#         -------
#         hatL: float
#             The squared loss, that is the sum of the squares of Y-Ypred, where Ypred are the fitted values of the Ridge
#             regression of Y against signatures of X truncated at m.
#         """
#         Ypred = self.predict(X)
#         if plot:
#             plt.scatter(Y, Ypred)
#             plt.plot([0.9 * np.min(Y), 1.1 * np.max(Y)], [0.9 * np.min(Y), 1.1 * np.max(Y)], '--', color='black')
#             plt.title("Ypred against Y")
#             plt.show()
#         return np.mean((Y - Ypred) ** 2)
    
#     def score(self, X,Y): ##added by Nik
#         return 1-self.get_loss(X,Y)/ np.mean((Y-np.mean(Y))**2)
    
    
class SignatureRegressionNik():
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

    def __init__(self, m, normalizeFeatures=False, alpha=None):
        self.normalizeFeatures = normalizeFeatures
        self.reg = Ridge(normalize=False, fit_intercept=False, solver='svd')
        self.m = m
        self.alpha = alpha
        if self.normalizeFeatures:
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

        sigX = get_sigX(X,self.m)
        self.sigX = sigX
        
        if self.normalizeFeatures:
            self.scaler.fit(sigX)
            sigX = self.scaler.transform(sigX)
            
        if self.alpha is None: #select alpha by cross-validation
            alphas=np.linspace(10 ** (-6), 100, num=1000)
            self.reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            self.reg_cv.fit(sigX, Y)
            self.alpha = self.reg_cv.alpha_
            
        self.reg = Ridge(alpha = self.alpha, fit_intercept=False)
        self.reg.fit(sigX,Y)
        
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
        if self.normalizeFeatures:
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
    
    def score(self, X,Y):
        return 1-self.get_loss(X,Y)/ np.mean((Y-np.mean(Y))**2)
    
    def fit_fromSig(self, sigX, Y, alphas=np.linspace(10 ** (-6), 100, num=1000)):
        
        if self.normalizeFeatures:
            self.scaler.fit(sigX)
            sigX = self.scaler.transform(sigX)
            
        if self.alpha is None: #select alpha by cross-validation
            self.reg_cv = RidgeCV(alphas=alphas, store_cv_values=True, fit_intercept=False, gcv_mode='svd')
            self.reg_cv.fit(sigX, Y)
            self.alpha = self.reg_cv.alpha_
            
        self.reg = Ridge(alpha = self.alpha, fit_intercept=False)
        self.reg.fit(sigX,Y)
        
        return self.reg
    
    def predict_fromSig(self, sigX):
        if self.normalizeFeatures:
            sigX = self.scaler.transform(sigX)
        Ypred = self.reg.predict(sigX)
        return Ypred
    
    def get_loss_fromSig(self, sigX, Y, plot = False):
        Ypred = self.predict_fromSig(sigX)
        return np.mean((Y - Ypred) ** 2)
    
    def score_fromSig(self, sigX, Y):
        return 1-self.get_loss_fromSig(sigX,Y)/ np.mean((Y-np.mean(Y))**2)
    
# if __name__ == '__main__':

#     import dataGeneration as dg
    
#     dimPath = 2
#     nPaths = 10000
#     mStar = 5
    
#     G = dg.GeneratorFermanian1(dimPath,nPaths,mStar, num = 101)
#     G.generatePath()
#     G.generateResponse()
    
#     #X = np.array(G.X)
    
#     # add time:
#     X = np.array([np.concatenate((G.partition01.reshape(-1,1), x),axis = 1) for x in G.X])
#     Y = G.Y
    
#     Kpen = getKpen(X,Y,max_Kpen = 2000,rho = 0.25,alpha = None,normalizeFeatures = True, plotTrue = True)
    
#     mHat, reg,_ = getmHat(X, Y, Kpen, rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = True)
    
#     print('Kpen: ', Kpen)
#     print('m_hat: ', mHat)
#     print('alpha: ', reg.alpha)
    
    