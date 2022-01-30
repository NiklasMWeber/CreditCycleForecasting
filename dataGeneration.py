# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:50:46 2022

@author: nikth
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
    
    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
    
        
    def generatePath(self):
        self.a = np.random.rand(self.nPaths,self.dimPath,4)
        self.X = np.array([self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)])
        #[self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)]
        #[self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2] for i in range(self.nPaths)]
        return self.X
        
    def generateResponse(self):
        self.sigX = get_sigX(self.X, self.mStar) #[ts.stream2sig(self.X[i],self.mStar) for i in range(self.nPaths)]
        self.sigdim = len(self.sigX[0])
        self.beta = np.random.rand(self.sigdim)/1000
        self.eps = np.random.uniform(-100,100,size=self.nPaths)
        self.Y = [np.dot(self.beta,self.sigX[i])+self.eps[i] for i in range(self.nPaths)]
        return np.array(self.Y)
    
    
class GeneratorFermanianIndependentMean(DataGenerator):
    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
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
    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
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
    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
    
        
    def generatePath(self):
        #self.a = np.random.rand(self.nPaths,self.dimPath,4)
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
    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
    
        
    def generatePath(self):
        #self.a = np.random.rand(self.nPaths,self.dimPath,4)
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

    def __init__(self, dimPath, nPaths, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
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

        #Y = np.sqrt(np.sum(slope ** 2, axis=1))
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
    
    def __init__(self, dimPath = None, nPaths = None, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        if num != None:
            self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
    
    def generatePath(self):
        x_1,x_2,x_3,x_4_1,x_4_2,y = importData()
        X,Y,year = prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y)
        del x_1,x_2,x_3,x_4_1,x_4_2,y

        # Standardize Data
        X = StandardScaler().fit_transform(X) #-mean() --> /std
        max_abs_scaler = MaxAbsScaler()
        Y_scaled = max_abs_scaler.fit_transform(Y-np.mean(Y))# -mean --> range [-1,1]

    # Construct 3 year rolling windows:
        reg_data = []
        predictors = []
        predictors_for_Signature = []
        for i in range(3,len(year)):
            predictors.append(X[(i-3):i,:].reshape(-1))  
            predictors_for_Signature.append(X[(i-3):i,:])
    
        predictors = np.array(predictors_for_Signature)
        self.X = predictors
        self.Y = np.array(Y_scaled[3:len(year)])
        return self.X
    
    def generateResponse(self):
        return self.Y
    
class GeneratorMacroDataFromNumpy(DataGenerator):
    
    def __init__(self, dimPath = None, nPaths = None, mStar = None, num = None):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        if num != None:
            self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.num = num+1
        self.partition01 = np.linspace(0,1,num=self.num)
    
    def generatePath(self):
        #x_1,x_2,x_3,x_4_1,x_4_2,y = importData()
        mat = np.load('macrodata.npy')
        X,Y,year = mat[:,1:-1], mat[:,-1].reshape((-1,1)), mat[:,0].reshape((-1,1))
        #del x_1,x_2,x_3,x_4_1,x_4_2,y

        # Standardize Data
        X = StandardScaler().fit_transform(X) #-mean() --> /std
        max_abs_scaler = MaxAbsScaler()
        Y_scaled = max_abs_scaler.fit_transform(Y-np.mean(Y))# -mean --> range [-1,1]

    # Construct 3 year rolling windows:
        reg_data = []
        predictors = []
        predictors_for_Signature = []
        for i in range(3,len(year)):
            predictors.append(X[(i-3):i,:].reshape(-1))  
            predictors_for_Signature.append(X[(i-3):i,:])
    
        predictors = np.array(predictors_for_Signature)
        self.X = predictors
        self.Y = np.array(Y_scaled[3:len(year)])
        return self.X
    
    def generateResponse(self):
        return self.Y
    
   
########## Classes to generate Fermanian Data (using her Method) ########################################
# from get_data import get_train_test_data    
# class GeneratorFermanianDependentMaxTest(DataGenerator):
#     def __init__(self, dimPath, nPaths, mStar = None, num = None):
#         DataGenerator.__init__(self)
#         self.dimPath = dimPath
#         self.nPaths = nPaths
#         self.num = num+1
#         self.partition01 = np.linspace(0,1,num=num)
#         self.mStar = mStar
        
#     def set_nPaths(self, nPaths):
#         self.nPaths = nPaths
        
#     def set_numForPartition(self,num):
#         self.num = num+1
#         self.partition01 = np.linspace(0,1,num=self.num)
    
        
#     def generatePath(self):
#         npoints = self.num
        
#         Xtrain, Ytrain, _ , _ = get_train_test_data('smooth_dependent', ntrain=self.nPaths, nval=1,  Y_type='max', npoints=npoints, d=self.dimPath, scale_X=False)

#         self.X = Xtrain
#         self.Y = Ytrain
#         return self.X
    
#     def generateResponse(self):
#         return self.Y
    
# class GeneratorFermanian1Test(DataGenerator):
#     # Always uses mStar = 5. Fermanian hardcoded it!
    
#     def __init__(self, dimPath, nPaths, mStar = None, num = None):
#         DataGenerator.__init__(self)
#         self.dimPath = dimPath
#         self.nPaths = nPaths
#         self.num = num+1
#         self.partition01 = np.linspace(0,1,num=num)
#         self.mStar = mStar
        
#     def set_nPaths(self, nPaths):
#         self.nPaths = nPaths
        
#     def set_numForPartition(self,num):
#         self.num = num+1
#         self.partition01 = np.linspace(0,1,num=self.num)
    
        
#     def generatePath(self):
#         npoints = self.num
        
#         Xtrain, Ytrain, _ , _ = get_train_test_data('smooth_independent', ntrain=self.nPaths, nval=1,  Y_type='sig', npoints=npoints, d=self.dimPath, scale_X=False)

#         self.X = Xtrain
#         self.Y = Ytrain
#         return self.X
    
#     def generateResponse(self):
#         return self.Y 
##############################################################################      
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dimPath = 5
    nPaths = 10
    num = 101
    mStar = 5
    
    mat = np.load('macrodata.npy')
    G = GeneratorMacroDataFromNumpy(dimPath = dimPath,nPaths = nPaths,mStar = mStar,num = num)
    G.generatePath()
    
    G2 = GeneratorMacroData(dimPath = dimPath,nPaths = nPaths,mStar = mStar,num = num)
    G2.generatePath()
    G2.generateResponse()
    
    X = G.X[0]
    #a = G.a[0,:,2]
    G.generateResponse()
    
    ### Some plotting
    plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,0], 'r')
    plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,1], 'b')
    plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,2], 'g')
    plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,3], 'y')
    #plt.plot(np.linspace(0,1,num = len(G.X[0,:,0])), G.X[0][:,4], 'orange')
    plt.show()
    
    ### Test how signature works --> time is No. of rows. Each column is one datatype e.g. GDP     
    #X = np.array([[1,1],[2,2],[1.5,3]])
    #sigX = ts.stream2sig(X, 2)
