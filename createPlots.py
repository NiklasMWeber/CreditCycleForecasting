# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 17:27:58 2022

@author: nikth
"""
import pandas as pd
import numpy as np
import os
from tools import plotTable
ListOfExperimentsWithSyntheticalData = ['exp1', 'exp2', 'exp3']
ListOfExperimentsWithRealData = ['exp4']
path = os.getcwd()
trueM = None

for nameForExperimentFolder in ListOfExperimentsWithSyntheticalData:
    mHatMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\mHatMean.txt', index_col = 0, header = 0)
    mHatStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\mHatStd.txt', index_col = 0, header = 0)
    R_SigTestMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_SigTestMean.txt', index_col = 0, header = 0)
    R_SigTestStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_SigTestStd.txt', index_col = 0, header = 0)
    R_LinRegTestMeanRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_LinRegTestMean.txt', index_col = 0, header = 0)
    R_LinRegTestStdRead = pd.read_csv('exp\\'+ nameForExperimentFolder + '\\R_LinRegTestStd.txt', index_col = 0, header = 0)
    
    numForPartitionList = mHatMeanRead.columns
    nPathsList =  mHatMeanRead.index
    
    mHat_Matrix = np.zeros((mHatMeanRead.shape[0],mHatMeanRead.shape[1],2))
    mHat_Matrix[:,:,0] = mHatMeanRead.values
    mHat_Matrix[:,:,1] = mHatStdRead.values
    R_Sig_testMatrix = np.zeros((R_SigTestMeanRead.shape[0],R_SigTestMeanRead.shape[1],2))
    R_Sig_testMatrix[:,:,0] = R_SigTestMeanRead.values
    R_Sig_testMatrix[:,:,1] = R_SigTestStdRead.values
    R_LinReg_testMatrix = np.zeros((R_LinRegTestMeanRead.shape[0],R_LinRegTestMeanRead.shape[1],2))
    R_LinReg_testMatrix[:,:,0] = R_LinRegTestMeanRead.values
    R_LinReg_testMatrix[:,:,1] = R_LinRegTestStdRead.values
    
    
    plotTable(data = mHat_Matrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'meanM', trueM = trueM)
    plotTable(data = mHat_Matrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'std', trueM = trueM)
    plotTable(data = R_Sig_testMatrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'meanR', trueM = trueM)
    plotTable(data = R_Sig_testMatrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'std', trueM = trueM)
    plotTable(data = R_LinReg_testMatrix[:,:,0], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'meanR', trueM = trueM)
    plotTable(data = R_LinReg_testMatrix[:,:,1], rowList = nPathsList,colList = numForPartitionList,
          colLabel = 'nPoints',rowLabel = 'nPaths', type = 'std', trueM = trueM)
    
for nameForExperimentFolder in ListOfExperimentsWithRealData:
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
    

    plotTable(data = mHat_Matrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanM', Qflag = True)
    plotTable(data = mHat_Matrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', Qflag = True)
    plotTable(data = R_Sig_testMatrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanR', Qflag = True)
    plotTable(data = R_Sig_testMatrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', Qflag = True)
    plotTable(data = R_LinReg_testMatrix[:,:,0], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'meanR', Qflag = True)
    plotTable(data = R_LinReg_testMatrix[:,:,1], rowList = windowSizes,colList = forecastingHorizon,
          colLabel = 'Forecasting Horizon',rowLabel = 'Window Size', type = 'std', Qflag = True)