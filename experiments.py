# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:50:48 2022

@author: nikth
"""

# class M_Hat_Experiment():
    
#     def __init__(self,X,Y,)


import numpy as np
import dataGeneration as dg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from train import get_sigX, getKpen, getmHat




class CompareSigAndLinReg():
    
    def __init__(self, X,Y, testRatio = 0.3):
        self.X = X
        self.Y = Y
        self.testRatio = testRatio
        
    def compare(self):
        
        #Find params for Signature Regression
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size = self.testRatio)
        self.Kpen = getKpen(self.X_train,self.Y_train,max_Kpen = 2000,rho = 0.25,alpha = None,normalizeFeatures = True, plotTrue = False)
        self.mHat, self.reg, self.scaler = getmHat(self.X_train, self.Y_train, self.Kpen, rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = False)
        
        #Compute Singatures and reshape inputs for both regressions
        self.sigX_test = get_sigX(self.X_test, self.mHat)
        self.sigX_train = get_sigX(self.X_train, self.mHat)
        if self.scaler != None:
            self.sigX_train = self.scaler.transform(self.sigX_train)
            self.sigX_test = self.scaler.transform(self.sigX_test)
            
        self.X_LinReg_train = self.X_train.reshape(len(self.X_train),-1)
        self.X_LinReg_test = self.X_test.reshape(len(self.X_test),-1)
        

        self.Y_pred_Sig_train = self.reg.predict(self.sigX_train)
        self.Y_pred_Sig_test = self.reg.predict(self.sigX_test)
        
        self.alphas=np.linspace(10 ** (-6), 100, num=1000)
        self.linReg_cv = RidgeCV(alphas=self.alphas, store_cv_values=False, fit_intercept=True, gcv_mode='svd')
        self.linReg_cv.fit(self.X_LinReg_train, self.Y_train)
        
        self.Y_pred_LinReg_train = self.linReg_cv.predict(self.X_LinReg_train)
        self.Y_pred_LinReg_test = self.linReg_cv.predict(self.X_LinReg_test)
        
        
        self.MSE_Sig_test = np.mean((self.Y_test-self.Y_pred_Sig_test)**2)
        self.MSE_Sig_train = np.mean((self.Y_train-self.Y_pred_Sig_train)**2)
        self.MSE_LinReg_test = np.mean((self.Y_test-self.Y_pred_LinReg_test)**2)
        self.MSE_LinReg_train = np.mean((self.Y_train-self.Y_pred_LinReg_train)**2)
        self.R_Sig_test = self.reg.score(self.sigX_test,self.Y_test)
        self.R_Sig_train = self.reg.score(self.sigX_train,self.Y_train)
        self.R_LinReg_test =self.linReg_cv.score(self.X_LinReg_test, self.Y_test)
        self.R_LinReg_train = self.linReg_cv.score(self.X_LinReg_train, self.Y_train)
        
        return self.MSE_Sig_test,self.MSE_LinReg_test,self.R_Sig_test,self.R_LinReg_test
    
    def createMatrix(self,nPathsList, numForPartitionList, iterations = None):
        return None
 