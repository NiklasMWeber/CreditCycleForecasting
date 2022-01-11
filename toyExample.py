# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:10:05 2022

@author: nikth
"""

import numpy as np
import dataGeneration as dg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt
from train import get_sigX, getKpen, getmHat, SignatureRegression
from experiments import CompareSigAndLinReg




# class CompareSigAndLinReg():
    
#     def __init__(self, X,Y, testRatio = 0.3):
#         self.X = X
#         self.Y = Y
#         self.testRatio = testRatio
        
#     def compare(self):
        
#         #Find params for Signature Regression
#         self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size = self.testRatio)
#         self.Kpen = getKpen(self.X_train,self.Y_train,max_Kpen = 2000,rho = 0.25,alpha = None,normalizeFeatures = True, plotTrue = False)
#         self.mHat, self.reg, self.scaler = getmHat(self.X_train, self.Y_train, self.Kpen, rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = False)
        
#         #Compute Singatures and reshape inputs for both regressions
#         self.sigX_test = get_sigX(self.X_test, self.mHat)
#         self.sigX_train = get_sigX(self.X_train, self.mHat)
#         if self.scaler != None:
#             self.sigX_train = self.scaler.transform(self.sigX_train)
#             self.sigX_test = self.scaler.transform(self.sigX_test)
            
#         self.X_LinReg_train = self.X_train.reshape(len(self.X_train),-1)
#         self.X_LinReg_test = self.X_test.reshape(len(self.X_test),-1)
        

#         self.Y_pred_Sig_train = self.reg.predict(self.sigX_train)
#         self.Y_pred_Sig_test = self.reg.predict(self.sigX_test)
        
#         self.alphas=np.linspace(10 ** (-6), 100, num=1000)
#         self.linReg_cv = RidgeCV(alphas=self.alphas, store_cv_values=False, fit_intercept=True, gcv_mode='svd')
#         self.linReg_cv.fit(self.X_LinReg_train, self.Y_train)
        
#         self.Y_pred_LinReg_train = self.linReg_cv.predict(self.X_LinReg_train)
#         self.Y_pred_LinReg_test = self.linReg_cv.predict(self.X_LinReg_test)
        
        
#         self.MSE_Sig_test = np.mean((self.Y_test-self.Y_pred_Sig_test)**2)
#         self.MSE_Sig_train = np.mean((self.Y_train-self.Y_pred_Sig_train)**2)
#         self.MSE_LinReg_test = np.mean((self.Y_test-self.Y_pred_LinReg_test)**2)
#         self.MSE_LinReg_train = np.mean((self.Y_train-self.Y_pred_LinReg_train)**2)
#         self.R_Sig_test = self.reg.score(self.sigX_test,self.Y_test)
#         self.R_Sig_train = self.reg.score(self.sigX_train,self.Y_train)
#         self.R_LinReg_test =self.linReg_cv.score(self.X_LinReg_test, self.Y_test)
#         self.R_LinReg_train = self.linReg_cv.score(self.X_LinReg_train, self.Y_train)
        
#         return self.MSE_Sig_test,self.MSE_LinReg_test,self.R_Sig_test,self.R_LinReg_test
    
#     def createMatrix(self,nPathsList, numForPartitionList, iterations = None):
#         return None
 
       
# Create Data
dimPath = 3
nPaths = 1000
partition01 = np.linspace(0,1,num=11)
mStar = 5

G = dg.GeneratorFermanian1(dimPath,nPaths,partition01,mStar)
G.generatePath()
G.generateResponse()

X = np.array(G.X)
Y = G.Y  

comparer = CompareSigAndLinReg(X,Y,testRatio = 0.3)
loss_sig,loss_linReg,R_sig,R_linReg =  comparer.compare()
print('new Class') 
print('Sig MSE: ', loss_sig, '\tR: ', R_sig)
print('LinReg MSE: ', loss_linReg, '\tR: ', R_linReg)
    
        
X_train, X_test, Y_train, Y_test = train_test_split( X
                                                    ,Y, test_size = 1/3)

Kpen = getKpen(X_train,Y_train,max_Kpen = 2000,rho = 0.25,alpha = None,normalizeFeatures = True, plotTrue = False)
mHat, reg, scaler = getmHat(X_train, Y_train, Kpen, rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = False)


sigX = get_sigX(X_test, mHat)
if scaler != None:
    sigX = scaler.transform(sigX)
    
Y_pred_Sig = reg.predict(sigX)

XLinReg_train, XLinReg_test = X_train.reshape(len(X_train),-1), X_test.reshape(len(X_test),-1)
alphas=np.linspace(10 ** (-6), 100, num=1000)
linReg_cv = RidgeCV(alphas=alphas, store_cv_values=False, fit_intercept=True, gcv_mode='svd')
linReg_cv.fit(XLinReg_train, Y_train)
Y_pred_LinReg = linReg_cv.predict(XLinReg_test)

error_sig = (Y_test-Y_pred_Sig)
loss_sig = np.mean(error_sig**2)

error_linReg = Y_test-Y_pred_LinReg
loss_linReg = np.mean(error_linReg**2)

# plt.figure()
# plt.plot(np.linspace(1,len(Y_pred_Sig), num =len(Y_pred_Sig)),Y_pred_Sig, 'r')
# plt.plot(np.linspace(1,len(Y_pred_Sig), num =len(Y_pred_Sig)),Y_test, 'b')
# plt.plot(np.linspace(1,len(Y_pred_Sig), num =len(Y_pred_Sig)),Y_pred_LinReg, 'orange')
print('old Class:')
print('Sig MSE: ', loss_sig, '\tR: ', reg.score(sigX, Y_test))
print('LinReg MSE: ', loss_linReg, '\tR: ', linReg_cv.score(XLinReg_test, Y_test))