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
    
    def __init__(self, X = None,Y = None, testRatio = 0.3):
        self.X = X
        self.Y = Y
        self.testRatio = testRatio
        
    def compare(self, X = None, Y = None):
        
        if X != None: self.X = X
        if Y != None: self.Y = Y
        
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
    
    def createComparisonMatrix(self,nPathsList, numForPartitionList,dataGenerator, iterations = 1):
        
        self.numErr = 0
        
        MSE_Sig_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2)) 
        MSE_LinReg_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        R_Sig_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        R_LinReg_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2)) 
        
        for i_nPaths, nPaths in enumerate(nPathsList):
            print('nPaths: ', nPaths)
            for j_num, num in enumerate(numForPartitionList):
                print('num: ', num)
                dataGenerator.set_nPaths(nPaths)
                dataGenerator.set_numForPartition(num)
                MSE_Sig_testL, MSE_LinReg_testL, R_Sig_testL, R_LinReg_testL = [],[],[],[]
                for i in range(1,iterations+1):
                    self.X = dataGenerator.generatePath()
                    self.Y = dataGenerator.generateResponse()
                    #calcSuccess = False
                    MSE_Sig_test, MSE_LinReg_test, R_Sig_test, R_LinReg_test = self.compare()
                    #while calcSuccess == False:
                        # try:
                        #     MSE_Sig_test, MSE_LinReg_test, R_Sig_test, R_LinReg_test = self.compare()
                        #     calcSuccess =True
                        # except:
                        #     self.numErr += 1
                        #     print('Calc Error No. ' + str(self.numErr))
                    MSE_Sig_testL.append(MSE_Sig_test)
                    MSE_LinReg_testL.append(MSE_LinReg_test), 
                    R_Sig_testL.append(R_Sig_test) 
                    R_LinReg_testL.append(R_LinReg_test)
                
                MSE_Sig_testMatrix[i_nPaths][j_num][0] = np.mean(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_nPaths][j_num][0] = np.mean(MSE_LinReg_testL)
                R_Sig_testMatrix[i_nPaths][j_num][0] = np.mean(R_Sig_testL)
                R_LinReg_testMatrix[i_nPaths][j_num][0] = np.mean(R_LinReg_testL)
                
                MSE_Sig_testMatrix[i_nPaths][j_num][1] = np.std(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_nPaths][j_num][1] = np.std(MSE_LinReg_testL)
                R_Sig_testMatrix[i_nPaths][j_num][1] = np.std(R_Sig_testL)
                R_LinReg_testMatrix[i_nPaths][j_num][1] = np.std(R_LinReg_testL)
                
        
        self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix, self.R_LinReg_testMatrix = MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix
        return self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix, self.R_LinReg_testMatrix
    
if __name__ == '__main__':
    comparer = CompareSigAndLinReg(testRatio = 0.3)
    
    dimPath = 3
    mStar = 5
    nPathsList = [33,50,100,200,500,1000,3000]
    numForPartitionList = [3,5,10,20,50,100]
    
    G = dg.GeneratorFermanian1(dimPath,nPathsList[0],mStar, num = numForPartitionList[0])
    
    MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix = comparer.createComparisonMatrix(nPathsList,numForPartitionList,G,20)
    
    
    


    
 