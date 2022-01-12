# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:50:46 2022

@author: nikth
"""
import numpy as np
import math
from esig import tosig as ts
####
import matplotlib.pyplot as plt

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
    def __init__(self, dimPath, nPaths, mStar, num):
        DataGenerator.__init__(self)
        self.dimPath = dimPath
        self.nPaths = nPaths
        self.num = num
        self.partition01 = np.linspace(0,1,num=num)
        self.mStar = mStar
        
    def set_nPaths(self, nPaths):
        self.nPaths = nPaths
        
    def set_numForPartition(self,num):
        self.partition01 = np.linspace(0,1,num=num)
    
        
    def generatePath(self):
        self.a = np.random.rand(self.nPaths,self.dimPath,4)
        self.X = [self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)]
        #[self.a[i,:,0]+10*self.a[i,:,1]*np.sin(2*math.pi*self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2]) +10*(self.partition01.reshape(len(self.partition01),1) - self.a[i,:,3])**3 for i in range(self.nPaths)]
        #[self.partition01.reshape(len(self.partition01),1) /self.a[i,:,2] for i in range(self.nPaths)]
        return np.array(self.X)
        
    def generateResponse(self):
        self.sigX = [ts.stream2sig(self.X[i],self.mStar) for i in range(self.nPaths)]
        self.sigdim = len(self.sigX[0])
        self.beta = np.random.rand(self.sigdim)/1000
        self.eps = np.random.uniform(-100,100,size=self.nPaths)
        self.Y = [np.dot(self.beta,self.sigX[i])+self.eps[i] for i in range(self.nPaths)]
        return np.array(self.Y)
        
       
if __name__ == '__main__':
    
    dimPath = 4
    nPaths = 100
    partition01 = np.array([j*0.01 for j in range(101)])
    mStar = 5
    
    G = GeneratorFermanian1(dimPath,nPaths,partition01,mStar)
    
    G.generatePath()
    X = G.X[0]
    a = G.a[0,:,2]
    G.generateResponse()
    
    ### Some plotting
    # plt.plot(G.partition01, G.X[0][:,0], 'r')
    # plt.plot(G.partition01, G.X[0][:,1], 'b')
    # plt.plot(G.partition01, G.X[0][:,2], 'g')
    # plt.plot(G.partition01, G.X[0][:,3], 'y')
    # plt.show()
    
    ### Test how signature works --> time is No. of rows. Each column is one datatype e.g. GDP     
    #X = np.array([[1,1],[2,2],[1.5,3]])
    #sigX = ts.stream2sig(X, 2)
