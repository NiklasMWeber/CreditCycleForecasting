# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:50:48 2022

@author: nikth
"""

# class M_Hat_Experiment():
#     def __init__(self,X,Y,)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from train import get_sigX, select_hatm_cv, SignatureRegressionNik
from tools import add_time, add_basepoint, plotTable


class CompareSigAndLinReg():
    
    def __init__(self, X = None,Y = None, testRatio = 0.3):
        self.X = X
        self.Y = Y
        self.testRatio = testRatio
        
    def compare(self, X = None, Y = None, Kpen = None, mHat = None, normalizeFeatures = True):
        
        if X != None: self.X = X
        if Y != None: self.Y = Y
        
        #Find params for Signature Regression
        self.X_train, self.X_test, self.Y_train, self.Y_test =         \
            train_test_split(self.X,self.Y,test_size = self.testRatio)
        
        #self.X[:80], self.X[80:], self.Y[:80], self.Y[80:]
            
        # if (Kpen == None) and (mHat == None): #dont need Kpen if mHat is given
        #     self.Kpen = getKpen(self.X_train,self.Y_train,max_Kpen = 2000,rho = 0.25,
        #                     alpha = None,normalizeFeatures = True, plotTrue = False)
        # else:
        #     self.Kpen = Kpen
        self.Kpen = Kpen
        
        
        if mHat == None:
            self.mHat = select_hatm_cv(self.X_train, self.Y_train, max_k=None, normalizeFeatures=True, plot=False)
        else:
            self.mHat = mHat 
            # , _ , _ = getmHat(self.X_train, self.Y_train,self.Kpen, 
            #      rho = 0.25, alpha = None, m_max = None, normalizeFeatures=True, plotTrue = False, mHatInput = mHat)
        
        ### Signature Regression:
        
        self.reg = SignatureRegressionNik(
            m = self.mHat, normalizeFeatures = normalizeFeatures, alpha = None)
        self.reg.fit(self.X_train,self.Y_train)
        
        self.sigX_train = self.reg.sigX
        self.sigX_test = get_sigX(X = self.X_test, m=self.mHat)
        
        self.Y_pred_Sig_train = self.reg.predict_fromSig(self.sigX_train)
        self.Y_pred_Sig_test = self.reg.predict_fromSig(self.sigX_test)
        
        
        ### Linear Regression:
        self.X_LinReg_train = self.X_train.reshape(len(self.X_train),-1)
        self.X_LinReg_test = self.X_test.reshape(len(self.X_test),-1)
        
        
        self.alphas=np.linspace(10 ** (-6), 100, num=1000)
        self.linReg_cv = RidgeCV(alphas=self.alphas, store_cv_values=False,
                                 fit_intercept=True, gcv_mode='svd')
        
        self.linReg_cv.fit(self.X_LinReg_train, self.Y_train)
        #self.alpha = self.linReg_cv.alpha_
        
        self.Y_pred_LinReg_train = self.linReg_cv.predict(self.X_LinReg_train)
        self.Y_pred_LinReg_test = self.linReg_cv.predict(self.X_LinReg_test)
        
        self.MSE_Sig_test = np.mean((self.Y_test-self.Y_pred_Sig_test)**2)
        self.MSE_Sig_train = np.mean((self.Y_train-self.Y_pred_Sig_train)**2)
        self.MSE_LinReg_test = np.mean((self.Y_test-self.Y_pred_LinReg_test)**2)
        self.MSE_LinReg_train = np.mean((self.Y_train-self.Y_pred_LinReg_train)**2)
        self.R_Sig_test = self.reg.score_fromSig(self.sigX_test,self.Y_test)
        self.R_Sig_train = self.reg.score_fromSig(self.sigX_train,self.Y_train)
        self.R_LinReg_test =self.linReg_cv.score(self.X_LinReg_test, self.Y_test)
        self.R_LinReg_train = self.linReg_cv.score(self.X_LinReg_train, self.Y_train)
        
        #testPerf = self.R_Sig_test
        return self.MSE_Sig_test,self.MSE_LinReg_test,self.R_Sig_test,self.R_LinReg_test
    
    def createComparisonMatrix(self,nPathsList,numForPartitionList,dataGenerator, iterations = 1,
                               Kpen = None, mHat = None, addTime = True, normalizeFeatures = True):
        
        self.numErr = 0
        
        MSE_Sig_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2)) 
        MSE_LinReg_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        R_Sig_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        R_LinReg_testMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        
        R_Sig_trainMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        R_LinReg_trainMatrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2)) 
        
        mHat_Matrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        Kpen_Matrix = np.zeros(shape = (len(nPathsList),len(numForPartitionList),2))
        
        
        for i_nPaths, nPaths in enumerate(nPathsList):
            print('nPaths: ', nPaths)
            for j_num, num in enumerate(numForPartitionList):
                print('num: ', num)
                dataGenerator.set_nPaths(nPaths)
                dataGenerator.set_numForPartition(num)
                MSE_Sig_testL, MSE_LinReg_testL, R_Sig_testL, R_LinReg_testL = [],[],[],[]
                
                R_Sig_trainL, R_LinReg_trainL = [],[]
                
                mHatL,KpenL = [],[]
                
                for i in range(1,iterations+1):
                    self.X = dataGenerator.generatePath()
                    self.Y = dataGenerator.generateResponse()
                    
                    if addTime == True:
                        self.X = add_time(self.X)
                        self.X = add_basepoint(self.X)
                        #a=5

                    MSE_Sig_test, MSE_LinReg_test, R_Sig_test, R_LinReg_test = \
                        self.compare(Kpen = Kpen,mHat= mHat, normalizeFeatures = normalizeFeatures)
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
                    
                    R_Sig_trainL.append(self.R_Sig_train)
                    R_LinReg_trainL.append(self.R_LinReg_train)
                    
                    mHatL.append(self.mHat)
                    KpenL.append(self.Kpen)
                
                MSE_Sig_testMatrix[i_nPaths][j_num][0] = np.mean(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_nPaths][j_num][0] = np.mean(MSE_LinReg_testL)
                R_Sig_testMatrix[i_nPaths][j_num][0] = np.mean(R_Sig_testL)
                R_LinReg_testMatrix[i_nPaths][j_num][0] = np.mean(R_LinReg_testL)
                
                MSE_Sig_testMatrix[i_nPaths][j_num][1] = np.std(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_nPaths][j_num][1] = np.std(MSE_LinReg_testL)
                R_Sig_testMatrix[i_nPaths][j_num][1] = np.std(R_Sig_testL)
                R_LinReg_testMatrix[i_nPaths][j_num][1] = np.std(R_LinReg_testL)
                
                R_Sig_trainMatrix[i_nPaths][j_num][0] = np.mean(R_Sig_trainL)
                R_LinReg_trainMatrix[i_nPaths][j_num][0] = np.mean(R_LinReg_trainL)
                R_Sig_trainMatrix[i_nPaths][j_num][1] = np.std(R_Sig_trainL)
                R_LinReg_trainMatrix[i_nPaths][j_num][1] = np.std(R_LinReg_trainL)
                
                mHat_Matrix[i_nPaths][j_num][0] = np.mean(mHatL)
                mHat_Matrix[i_nPaths][j_num][1] = np.std(mHatL)
                Kpen_Matrix[i_nPaths][j_num][0] = np.mean(KpenL)
                Kpen_Matrix[i_nPaths][j_num][1] = np.std(KpenL)
                #mMax_Matrix[i_nPaths][j_num][0] = np.mean(mMaxL)
                #mMax_Matrix[i_nPaths][j_num][1] = np.std(mMaxL)
                
        self.R_Sig_trainMatrix, self.R_LinReg_trainMatrix = R_Sig_trainMatrix, R_LinReg_trainMatrix
        self.mHat_Matrix, self.Kpen_Matrix = mHat_Matrix, Kpen_Matrix
        #self.mMax_Matrix = mMax_Matrix
        
        self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix,\
            self.R_LinReg_testMatrix = MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix
        return self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix, self.R_LinReg_testMatrix
    
if __name__ == '__main__':
    import dataGeneration as dg
    
    comparer = CompareSigAndLinReg(testRatio = 0.33)
    
    nPathsList = [1]#33, 50, 100, 200, 500, 1000]
    numForPartitionList = [1]#[3,5,10,20,50,100]
    
    G = dg.GeneratorMacroDataFromNumpy(dimPath = 3, nPaths = nPathsList[0],mStar = 5, num = numForPartitionList[0])
    #G = dg.GeneratorMacroDataQ(windowSize = 3, forecastGap = 0)
    
    MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix = \
        comparer.createComparisonMatrix(nPathsList,numForPartitionList,G, iterations= 1, Kpen = 1, mHat = None, addTime = True)
    R_Sig_trainMatrix, R_LinReg_trainMatrix = comparer.R_Sig_trainMatrix, comparer.R_LinReg_trainMatrix
    mHat_Matrix = comparer.mHat_Matrix
    
    ###plotting generated data
    # plotTable(data = mHat_Matrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'meanM')
    # plotTable(data = mHat_Matrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'std')
    # plotTable(data = R_Sig_testMatrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'meanR')
    # plotTable(data = R_Sig_testMatrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'std')
    # plotTable(data = R_LinReg_testMatrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'meanR')
    # plotTable(data = R_LinReg_testMatrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
    #           colName = 'nPoints',rowName = 'nPaths', type = 'std')
    

    
    
    
    
    
    
    
    