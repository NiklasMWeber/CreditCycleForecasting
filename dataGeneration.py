# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:50:46 2022

This file provides classes that return different types of data, either synthetically generated or using the real-world macroeconomic data from the macrodata.npy array.

@author: Niklas Weber
"""

import numpy as np
import math
from tools import importData, prepareData
from train import get_sigX
from skfda.misc.covariances import Exponential
from skfda.datasets import make_gaussian_process
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

class DataGenerator:
    def __init__(self, generator= None):
        self.generator = generator
        self.X = None
        self.Y = None
        
    def generatePath(self):
        self.X = self.generator.generatePath()
        return self.X
        
    def generateResponse(self):
        self.Y = self.generator.generateResponse()
        return self.Y
              
class GeneratorFermanian1(DataGenerator):
    
    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
    
    def generatePath(self):
        self.a = np.random.rand(self.nPaths,self.dimPath,4)
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])

        return self.X
        
    def generateResponse(self):
        self.sigX = get_sigX(self.X, self.trueM)
        self.sigdim = len(self.sigX[0])
        self.beta = np.random.rand(self.sigdim)/1000
        self.eps = np.random.uniform(-100,100,size=self.nPaths)
        self.Y = [np.dot(self.beta,self.sigX[i])+self.eps[i] for i in range(self.nPaths)]
        return np.array(self.Y)
       
class GeneratorFermanianIndependentMean(DataGenerator):
    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
           
    def generatePath(self):
        self.a = np.random.rand(self.nPaths,self.dimPath,4)
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])       
        self.X_LastTimeIndex = self.X[:,-1,:]
        self.X = self.X[:,:-1,:]
        return self.X
        
    def generateResponse(self):
        
        self.Y = np.mean(self.X_LastTimeIndex, axis = 1)
        return np.array(self.Y)
    
class GeneratorFermanianIndependentMax(DataGenerator):
    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
           
    def generatePath(self):
        self.a = np.random.rand(self.nPaths,self.dimPath,4)
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])       
        self.X_LastTimeIndex = self.X[:,-1,:]
        self.X = self.X[:,:-1,:]
        return self.X
        
    def generateResponse(self):
        
        self.Y = np.max(self.X_LastTimeIndex, axis = 1)
        return np.array(self.Y)
       
class GeneratorFermanianDependentMean(DataGenerator):
    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
           
    def generatePath(self):
        self.a_path = np.random.rand(self.nPaths,1,4)
        self.a_dim =np.random.rand(1,self.dimPath,1) 
        self.a = self.a_path * self.a_dim
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])       
        self.X_LastTimeIndex = self.X[:,-1,:]
        self.X = self.X[:,:-1,:]
        return self.X
    
    def generateResponse(self):
        
        self.Y = np.mean(self.X_LastTimeIndex, axis = 1)
        return np.array(self.Y)
      
class GeneratorFermanianDependentMax(DataGenerator):
    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
         
    def generatePath(self):
        self.a_path = np.random.rand(self.nPaths,1,4)
        self.a_dim =np.random.rand(1,self.dimPath,1) 
        self.a = self.a_path * self.a_dim
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])       
        self.X_LastTimeIndex = self.X[:,-1,:]
        self.X = self.X[:,:-1,:]
        return self.X
    
    def generateResponse(self):
        
        self.Y = np.max(self.X_LastTimeIndex, axis = 1)
        return np.array(self.Y)
            
class GeneratorFermanianGaussian(DataGenerator):

    def __init__(self, dimPath, nPaths, trueM = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
    
    def generatePath(self):
        """ Generates n gaussian processes with a linear trend

		Parameters
		----------
		n: int
			Number of samples.

		Returns
		-------
		X: array, shape (n, self.npoints, self.d)
			Array of sample paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
			linear paths, each composed of npoints.
		"""
        X = np.zeros((self.nPaths, self.num, self.dimPath))
        times = np.repeat(np.expand_dims(np.linspace(0, 1, self.num), -1), self.nPaths * self.dimPath, 1)
        times = times.reshape((self.num, self.nPaths, self.dimPath)).transpose((1, 0, 2))
        self.slope = 3 * (2 * np.random.random((self.nPaths, self.dimPath)) - 1)

        slope = np.repeat(np.expand_dims(self.slope, 0), self.num, 0).transpose((1, 0, 2))
        for i in range(self.nPaths):
            gp = make_gaussian_process(n_features=self.num, n_samples=self.dimPath, cov=Exponential())
            X[i, :, :] = gp.data_matrix.T[0]

        self.X = X + slope * times
        return self.X
    
    def generateResponse(self):
        self.Y = np.sqrt(np.sum(self.slope ** 2, axis=1))
        return self.Y
       
class GeneratorMacroData(DataGenerator):
    
    def __init__(self, dimPath = None, nPaths = None, trueM = None, num = None,
                 windowSize = None, forecastGap = 0):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.windowSize = windowSize
        self.forecastGap = forecastGap
        if num != None:
            self.partition01 = np.linspace(0,1,num=num)
        self.trueM = trueM
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num
        self.partition01 = np.linspace(0,1,num=self.num)
        
    def set_windowSize(self, newSize):
        self.windowSize = newSize
        
    def set_forecastGap(self, newGap):
        self.forecastGap = newGap
    
    def generatePath(self):
        try:
            x_1,x_2,x_3,x_4_1,x_4_2,y = importData()
            X,Y,year = prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y)
            del x_1,x_2,x_3,x_4_1,x_4_2,y
        except:
            mat = np.load('macrodata.npy')
            X,Y = mat[:,1:-1].astype(float), mat[:,-1].reshape((-1,1)).astype(float)
        
        
        Y_scaledOld = MaxAbsScaler().fit_transform(Y-np.mean(Y)) 

        # Standardize Data
        X = StandardScaler().fit_transform(X) #-mean() --> /std
        Y_scaled = Y_scaledOld

    # Construct 3 year rolling windows:

        predictors_for_Signature = []
        for i in range(self.windowSize,len(Y)):
            
            XaugY = np.concatenate((X[(i-self.windowSize):i,:],Y_scaled[(i-self.windowSize):i,:]),axis = 1)
            predictors_for_Signature.append(XaugY)
    
        predictors_for_Signature = np.array(predictors_for_Signature)
        if self.forecastGap == 0:
            self.X = predictors_for_Signature #-0 = 0, therefore [:-0] would be empty slice...
        else:
            self.X = predictors_for_Signature[:(-self.forecastGap),:,:]
            
        self.Y = np.array(Y_scaled[self.windowSize+self.forecastGap:])

        return self.X
    
    def generateResponse(self):
        return self.Y
          
# if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    
    # dimPath = 5
    # nPaths = 10
    # num = 101
    # trueM = 5
    
    # G = GeneratorMacroData(windowSize = 12, forecastGap = 0)
    # G.generatePath()
    # G.generateResponse()
    
    
    # ### Some plotting
    # plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,0], 'r',label = 'GDP growth')
    # plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,1], 'b',label = 'Unemployment')
    # plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,2], 'g',label = 'S&P 500 growth')
    # plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,3], 'y',label = 'IR spread')
    # plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,4], 'orange', label = 'Lagged PDs')
    # plt.legend()
    # plt.show()
    
