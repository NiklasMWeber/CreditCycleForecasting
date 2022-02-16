# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:50:48 2022

This script contains a similar class as the `experiments.py` script. 
It can be used to perform simulations with the macroeconomic data and the varied parameters are different from the simulation with synthetically generated data.

@author: Niklas Weber
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from train import get_sigX, select_hatm_cv, SignatureRegression
from tools import add_time, add_basepoint, plotTable
import dataGeneration as dg

class CompareSigAndLinReg():
    
    def __init__(self, X = None,Y = None, testRatio = 0.3):
        self.X = X
        self.Y = Y
        self.testRatio = testRatio
        
    def compare(self, X = None, Y = None, mHat = None, normalizeFeatures = True, addTime = True, addBase = True):
        """ 
        This method performs one single comparison between Singature regression and linear regression .
        Results are stored in different arrays, some of which are returne, others are just stored as fields of the comparer.
        """
        if X != None: self.X = X
        if Y != None: self.Y = Y
        
        
        #Split Data into Train and Test set
        self.X_train, self.X_test, self.Y_train, self.Y_test =  \
            train_test_split(self.X,self.Y,test_size = self.testRatio)
        
        ### Augment paths for Signature Regression (add Basepoint 0 and time dimension):
        if addTime == True:
            self.X_train, self.X_test = add_time(self.X_train), add_time(self.X_test)
        if addBase == True:
            self.X_train, self.X_test = add_basepoint(self.X_train), add_basepoint(self.X_test)

        ### Get mHat
        if mHat == None:
            self.mHat = select_hatm_cv(self.X_train, self.Y_train, max_k=2, normalizeFeatures=True,
                                       plot=False)
        else:
            self.mHat = mHat 
            
        ### Signature Regression:
        self.reg = SignatureRegression(
            m = self.mHat, normalizeFeatures = normalizeFeatures, alpha = None)
        self.reg.fit(self.X_train,self.Y_train)
        
        # Save beta of Signature Regression
        self.sig_coef = self.reg.reg.coef_
        
        # Get Signature of train (already calc by reg.) and test set. Saves comp. time!
        self.sigX_train = self.reg.sigX
        self.sigX_test = get_sigX(X = self.X_test, m=self.mHat)
        
        ### Linear Regression:
        self.X_LinReg_train = self.X_train.reshape(len(self.X_train),-1)
        self.X_LinReg_test = self.X_test.reshape(len(self.X_test),-1)

        self.alphas=np.linspace(10 ** (-6), 100, num=1000)
        self.linReg_cv = RidgeCV(alphas=self.alphas, store_cv_values=False,
                                 fit_intercept=True, gcv_mode='svd')
        
        self.linReg_cv.fit(self.X_LinReg_train, self.Y_train)
        
        # Save beta of Linear Regression
        self.linReg_coef = self.linReg_cv.coef_
         
        ### Predict
        #Sig
        self.Y_pred_Sig_train = self.reg.predict_fromSig(self.sigX_train)
        self.Y_pred_Sig_test = self.reg.predict_fromSig(self.sigX_test)
        
        #Lin Reg
        self.Y_pred_LinReg_train = self.linReg_cv.predict(self.X_LinReg_train)
        self.Y_pred_LinReg_test = self.linReg_cv.predict(self.X_LinReg_test)
        
        ### Measure performance (MSE,R^2) for both regressions on Train and Test set.
        self.MSE_Sig_test = np.mean((self.Y_test-self.Y_pred_Sig_test)**2)
        self.MSE_Sig_train = np.mean((self.Y_train-self.Y_pred_Sig_train)**2)
        self.MSE_LinReg_test = np.mean((self.Y_test-self.Y_pred_LinReg_test)**2)
        self.MSE_LinReg_train = np.mean((self.Y_train-self.Y_pred_LinReg_train)**2)
        self.R_Sig_test = self.reg.score_fromSig(self.sigX_test,self.Y_test)
        self.R_Sig_train = self.reg.score_fromSig(self.sigX_train,self.Y_train)
        self.R_LinReg_test =self.linReg_cv.score(self.X_LinReg_test, self.Y_test)
        self.R_LinReg_train = self.linReg_cv.score(self.X_LinReg_train, self.Y_train)
        
        return self.MSE_Sig_test,self.MSE_LinReg_test,self.R_Sig_test,self.R_LinReg_test
    
    def createComparisonMatrix(self, windowSizes, forecastingGaps, dataGenerator,
                               iterations = 1, mHat = None,
                               normalizeFeatures = True, addTime = True, addBase = True):
        """ 
        This method performs multiple comparisons between Singature regression and linear regression.
        It repeatedly calls the 'compare' method for all combinations of 'nPath' and 'num'.
        Results are stored in different arrays, some of which are returne, others are just stored as fields of the comparer.
        """
        MSE_Sig_testMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2)) 
        MSE_LinReg_testMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2))
        R_Sig_testMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2))
        R_LinReg_testMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2))
        
        R_Sig_trainMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2))
        R_LinReg_trainMatrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2)) 
        
        mHat_Matrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps),2))
        
        coef_LinReg_Matrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps)), dtype=list)
        coef_Sig_Matrix = np.zeros(shape = (len(windowSizes),len(forecastingGaps)), dtype=list)
        
        for i_windowSize, windowSize in enumerate(windowSizes):
            print('WindowSize: ', windowSize)
            for j_gap, forecastingGap in enumerate(forecastingGaps):
                print('ForecastingGap: ', forecastingGap)
                dataGenerator.set_windowSize(windowSize)
                dataGenerator.set_forecastGap(forecastingGap)
                MSE_Sig_testL, MSE_LinReg_testL, R_Sig_testL, R_LinReg_testL = [],[],[],[]
                
                R_Sig_trainL, R_LinReg_trainL = [],[]
                
                mHatL = []
                
                coef_SigL = []
                coef_LinRegL = []
                
                for i in range(1,iterations+1):
                    self.X = dataGenerator.generatePath()
                    self.Y = dataGenerator.generateResponse()
                    
                    MSE_Sig_test, MSE_LinReg_test, R_Sig_test, R_LinReg_test = \
                        self.compare(mHat= mHat, normalizeFeatures = normalizeFeatures, addTime = addTime, addBase = addBase)
                    
                    MSE_Sig_testL.append(MSE_Sig_test)
                    MSE_LinReg_testL.append(MSE_LinReg_test), 
                    R_Sig_testL.append(R_Sig_test) 
                    R_LinReg_testL.append(R_LinReg_test)
                    
                    R_Sig_trainL.append(self.R_Sig_train)
                    R_LinReg_trainL.append(self.R_LinReg_train)
                    
                    mHatL.append(self.mHat)
                    coef_LinRegL.append(self.linReg_coef)
                    coef_SigL.append(self.sig_coef)
                
                MSE_Sig_testMatrix[i_windowSize][j_gap][0] = np.mean(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_windowSize][j_gap][0] = np.mean(MSE_LinReg_testL)
                R_Sig_testMatrix[i_windowSize][j_gap][0] = np.mean(R_Sig_testL)
                R_LinReg_testMatrix[i_windowSize][j_gap][0] = np.mean(R_LinReg_testL)
                
                MSE_Sig_testMatrix[i_windowSize][j_gap][1] = np.std(MSE_Sig_testL)
                MSE_LinReg_testMatrix[i_windowSize][j_gap][1] = np.std(MSE_LinReg_testL)
                R_Sig_testMatrix[i_windowSize][j_gap][1] = np.std(R_Sig_testL)
                R_LinReg_testMatrix[i_windowSize][j_gap][1] = np.std(R_LinReg_testL)
                
                R_Sig_trainMatrix[i_windowSize][j_gap][0] = np.mean(R_Sig_trainL)
                R_LinReg_trainMatrix[i_windowSize][j_gap][0] = np.mean(R_LinReg_trainL)
                R_Sig_trainMatrix[i_windowSize][j_gap][1] = np.std(R_Sig_trainL)
                R_LinReg_trainMatrix[i_windowSize][j_gap][1] = np.std(R_LinReg_trainL)
                
                mHat_Matrix[i_windowSize][j_gap][0] = np.mean(mHatL)
                mHat_Matrix[i_windowSize][j_gap][1] = np.std(mHatL)
                
                coef_LinReg_Matrix[i_windowSize][j_gap] = coef_LinRegL
                coef_Sig_Matrix[i_windowSize][j_gap] = coef_SigL

        self.R_Sig_trainMatrix, self.R_LinReg_trainMatrix = R_Sig_trainMatrix, R_LinReg_trainMatrix
        self.mHat_Matrix= mHat_Matrix
        self.coef_LinReg_Matrix, self.coef_Sig_Matrix = coef_LinReg_Matrix, coef_Sig_Matrix
        
        self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix,\
            self.R_LinReg_testMatrix = MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix
        return self.MSE_Sig_testMatrix, self.MSE_LinReg_testMatrix, self.R_Sig_testMatrix, self.R_LinReg_testMatrix
    
if __name__ == '__main__':
    
    ##########################################################################
    ### Specify experiment configuration
    nameForExperimentFolder = 'exp4'
    comparer = CompareSigAndLinReg(testRatio = 0.33)

    windowSizes = [3,4,6,8,12,16]
    forecastingGaps = [0,1,2,3,4,5,7,9,11,15,19,23]
    trueM = None
    plotTrue = False
    iterations = 20
    
    G = dg.GeneratorMacroData(windowSize = 3, forecastGap = 0)
    
    MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix = \
        comparer.createComparisonMatrix(windowSizes = windowSizes, forecastingGaps = forecastingGaps
                                        , dataGenerator = G, iterations = iterations, mHat = None,
                                        addTime = True, addBase = True)

    mHat_Matrix = comparer.mHat_Matrix
    
    ##########################################################################
    
    ### Check and create directories for all experiments (parent directory) and for this specific experiment 
    path = os.getcwd()
    if not os.path.isdir(path+'\exp'):
        os.mkdir(path+'\exp')
       
    while os.path.isdir(path+'\exp\\'+ nameForExperimentFolder) == True:
        nameForExperimentFolder += 'New'
        print('Directory with this name already existed. Appended \'New\' to the name.')
        
    foldername = path+'\exp\\'+ nameForExperimentFolder
    os.mkdir(foldername)
    
    ###Export results
    columns = [str(x) for x in forecastingGaps]
    index = [str(x) for x in windowSizes]
    
    mHatMean = pd.DataFrame(mHat_Matrix[:,:,0], columns = columns, index = index)
    mHatStd = pd.DataFrame(mHat_Matrix[:,:,1], columns = columns, index = index)
    R_SigTestMean  = pd.DataFrame(R_Sig_testMatrix[:,:,0], columns = columns, index = index)
    R_SigTestStd  = pd.DataFrame(R_Sig_testMatrix[:,:,1], columns = columns, index = index)
    R_LinRegTestMean  = pd.DataFrame(R_LinReg_testMatrix[:,:,0], columns = columns, index = index)
    R_LinRegTestStd  = pd.DataFrame(R_LinReg_testMatrix[:,:,1], columns = columns, index = index)
    
    mHatMean.to_csv(foldername + '\mHatMean.txt', index = True, header = True )
    mHatStd.to_csv(foldername + '\mHatStd.txt', index = True, header = True )
    R_SigTestMean.to_csv(foldername + '\R_SigTestMean.txt', index = True, header = True )
    R_SigTestStd.to_csv(foldername + '\R_SigTestStd.txt', index = True, header = True )
    R_LinRegTestMean.to_csv(foldername + '\R_LinRegTestMean.txt', index = True, header = True )
    R_LinRegTestStd.to_csv(foldername + '\R_LinRegTestStd.txt', index = True, header = True )   
    
    ###Plotting results
    if plotTrue == True:
        
        mHatMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\mHatMean.txt', index_col = 0, header = 0)
        mHatStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\mHatStd.txt', index_col = 0, header = 0)
        R_SigTestMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_SigTestMean.txt', index_col = 0, header = 0)
        R_SigTestStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_SigTestStd.txt', index_col = 0, header = 0)
        R_LinRegTestMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_LinRegTestMean.txt', index_col = 0, header = 0)
        R_LinRegTestStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_LinRegTestStd.txt', index_col = 0, header = 0)
        
        forecastingGaps = mHatMeanRead.columns
        windowSizes =  mHatMeanRead.index
        forecastingHorizon = [int(x)+1 for x in forecastingGaps]
        
        mHat_Matrix = np.zeros((mHatMeanRead.shape[0],mHatMeanRead.shape[1],2))
        mHat_Matrix[:,:,0] = mHatMeanRead.values
        mHat_Matrix[:,:,1] = mHatStdRead.values
        R_Sig_testMatrix = np.zeros((R_SigTestMeanRead.shape[0],R_SigTestMeanRead.shape[1],2))
        R_Sig_testMatrix[:,:,0] = R_SigTestMeanRead.values
        R_Sig_testMatrix[:,:,1] = R_SigTestStdRead.values
        R_LinReg_testMatrix = np.zeros((R_LinRegTestMeanRead.shape[0],R_LinRegTestMeanRead.shape[1],2))
        R_LinReg_testMatrix[:,:,0] = R_LinRegTestMeanRead.values
        R_LinReg_testMatrix[:,:,1] = R_LinRegTestStdRead.values
        
        out_filename =  path + '\\exp\\'+ nameForExperimentFolder + '\\config.py'
        with open(__file__, 'r') as f:
            with open(out_filename, 'w') as out:
                for line in (f.readlines()):
                    print(line, end='', file=out)
        
        plotTable(data = mHat_Matrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanM', MacroFlag = True)
        plotTable(data = mHat_Matrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', MacroFlag = True)
        plotTable(data = R_Sig_testMatrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanR', MacroFlag = True)
        plotTable(data = R_Sig_testMatrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', MacroFlag = True)
        plotTable(data = R_LinReg_testMatrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanR', MacroFlag = True)
        plotTable(data = R_LinReg_testMatrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
              colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', MacroFlag = True)
        
    ### Analyze reg coefficients (hardcoded!)
    
    # coef_LinReg_Matrix, coef_Sig_Matrix = comparer.coef_LinReg_Matrix, comparer.coef_Sig_Matrix

    # beta_5_0Sig = coef_Sig_Matrix[5,0]
    # mList = [x.shape[1] for x in beta_5_0Sig]
    # mList = list(dict.fromkeys(mList))
    # mList.sort()
    
    # betasSig_5_0 = []
    # for n in mList:
    #     beta = [x for x in beta_5_0Sig if x.shape[1] == n]
    #     beta = np.array(beta)
    #     betaMean = np.mean(beta, axis = 0).reshape(1,-1)
    #     betaStd = np.std(beta, axis=0).reshape(1,-1)
    #     betasSig_5_0.append(np.concatenate((betaMean,betaStd),axis=0))
        
    # beta_5_6Sig = coef_Sig_Matrix[5,6]
    # mList = [x.shape[1] for x in beta_5_6Sig]
    # mList = list(dict.fromkeys(mList))
    # mList.sort()
    
    # betasSig_5_6 = []
    # for n in mList:
    #     beta = [x for x in beta_5_6Sig if x.shape[1] == n]
    #     beta = np.array(beta)
    #     betaMean = np.mean(beta, axis = 0).reshape(1,-1)
    #     betaStd = np.std(beta, axis=0).reshape(1,-1)
    #     betasSig_5_6.append(np.concatenate((betaMean,betaStd),axis=0))
    
    # dimTrain = comparer.X_train.shape[2]
    # beta_5_0LinReg = np.mean(np.array(coef_LinReg_Matrix[5,0]).reshape(len(coef_LinReg_Matrix[5,0]),-1),axis=0
    #                          ).reshape(-1,dimTrain)
       
    # ### Plot beta coefficients (hardcoded!) Not flexible 

    # plt.figure()
    # x = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(1,1)', '(1,2)', '(1,3)', '(1,4)', '(1,5)',
    #  '(1,6)', '(2,1)', '(2,2)', '(2,3)', '(2,4)', '(2,5)', '(2,6)', '(3,1)', '(3,2)', '(3,3)', '(3,4)',
    #   '(3,5)', '(3,6)', '(4,1)', '(4,2)', '(4,3)', '(4,4)', '(4,5)', '(4,6)', '(5,1)', '(5,2)', '(5,3)',
    #    '(5,4)', '(5,5)', '(5,6)', '(6,1)', '(6,2)', '(6,3)', '(6,4)', '(6,5)', '(6,6)']
    
    # a1 = betasSig_5_0[0][0,1:].tolist()
    # a2 = betasSig_5_6[1][0,1:].tolist()
    # a1 += [0]*(len(a2)-len(a1))
    
    # maxx = max(max(a1),max(a2))
    # minx = min(min(a1),min(a2))
    # #,abs(min(a1)),abs(min(a2)))
    # #a3 = [100, 110, 120, 130, 115, 110]
    
    # df = pd.DataFrame(index=x, data={'Forecast Horizon t+1': a1,'Forecast Horizon t+8': a2})
    
    # plt.rcParams['font.size'] = '20'
    # fig, ax = plt.subplots(figsize=(10,25))

    # plott = df.plot(fontsize = 20, kind = 'barh', ax=ax, width=1, color = ['indianred','slateblue'])
    # plott.set_facecolor('lavender')
    # plott.legend(fontsize=20)
    # plt.show()  
    
    # ### Plot beta LinReg
    # plt.figure()
    # x = np.linspace(0,1,num = beta_5_0LinReg.shape[0])
    # plt.plot(x,beta_5_0LinReg[:,0], color = 'red', label = 'GDP growth')
    # plt.plot(x,beta_5_0LinReg[:,1], color = 'blue', label = 'Unemployment')
    # plt.plot(x,beta_5_0LinReg[:,2], color = 'green', label = 'S&P 500 growth')
    # plt.plot(x,beta_5_0LinReg[:,3], color = 'yellow', label = 'IR spread')
    # plt.plot(x,beta_5_0LinReg[:,4], color = 'orange', label = 'Lagged PDs')
    # plt.plot(x,beta_5_0LinReg[:,5], color = 'black', label = 'Time coordinate')
    # plt.legend(loc='lower left')
    # plt.ylabel('Coefficient linear regression')
    # plt.show()
           
    # del betaMean, beta
    