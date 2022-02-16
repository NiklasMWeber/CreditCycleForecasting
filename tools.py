# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:40:30 2022

This file contains some auxiliary functions that are used in other files.

@author: Niklas Weber
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def add_time(X):
	"""Adds a dimension with time to each smaple in X

	Parameters
	----------
	X: array, shape (n, npoints, d)
		Array of paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
		linear paths, each composed of n_points.

	Returns
	-------
	Xtime: array, shape (n, npoints, d + 1)
		Same array as X but with an extra dimension at the end, corresponding to time.
	"""
	times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))
	Xtime = np.concatenate([X, times.reshape((times.shape[0], times.shape[1], 1))], axis=2)
	return Xtime

def add_basepoint(X, bpoint = None): 
    """Adds a basepoint to each smaple in X

	Parameters
	----------
	X: array, shape (n, npoints, d)
		Array of paths. It is a 3-dimensional array, containing the coordinates in R^d of n piecewise
		linear paths, each composed of n_points.
        
    bpoint: array, shape (1,d)
        Basepoint. Default value 0

	Returns
	-------
	Xbase: array, shape (n, npoints+1, d)
		Same array as X but with an extra entry at the beginning - the basepoint.
	"""
    if bpoint == None:
        bpoint = np.tile(np.zeros(shape = (1,X.shape[2])), (X.shape[0], 1))
    Xbased = np.concatenate([bpoint.reshape((X.shape[0], 1, bpoint.shape[1])),X], axis=1)
    return Xbased
  
def importFile(fileWithPath, delimiter):
    """ 
    Simple importing of a file, line by line. 
    """
    x = []
    with open(fileWithPath, 'r') as file:
        file = csv.reader(file, delimiter=delimiter)
        for row in file:
            x.append(row)
    return x

def importData():
    """ 
    Imports data from the different paths using 'importFile()'
    """
    
    gdpPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\GDP.csv'
    unemplPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\Unemployment.csv'
    indexPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\SP500.txt'
    stirPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\shortTerm.csv'
    ltirPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\longTerm.csv'
    pdPath = r'C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data\PD_NORTH AMERICA.csv'
    
    x_1 = importFile(gdpPath, delimiter = ',')
    x_2 = importFile(unemplPath, delimiter = ',')
    x_3 = importFile(indexPath, delimiter = '\t')
    x_4_1 = importFile(stirPath, delimiter = ',')
    x_4_2 = importFile(ltirPath, delimiter = ',')
    y = importFile(pdPath, delimiter = ',')
    
    return x_1,x_2,x_3,x_4_1,x_4_2,y

def prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y):
    """
    Prepares data importet by 'ImportData()' by filtering out the correct years
    and dropping unecessary information.
    """
    
    y_min = 1990
    y_max = 2021
    
    #just filter for right years
    x_1 = np.array([x[1] for x in x_1[1:] if (float(x[0][:4])>=y_min and float(x[0][:4])<=y_max)]).astype(np.float64)
    x_1 = (x_1[1:]-x_1[:-1])/x_1[:-1]
    
    #just filter for right years
    x_2 = np.array([x[1] for x in x_2[1:] if (float(x[0][:4])>=y_min and float(x[0][:4])<=y_max)]).astype(np.float64)
    x_2 = x_2[1:]    
    
    ##extract dates -> average years -> take right years
    x_3 = np.array([x[5].replace(',','') for x in x_3[1:] 
                   if ((x[0][:6] in {'Dec 31','Jan 01', 'Mar 31', 'Apr 01', 'Jun 30', 'Jul 01', 'Sep 30', 'Oct 01'}) and (float(x[0][8:12]) <= y_max and float(x[0][8:12])>=y_min)) ]).astype(np.float64)
    x_3 = np.flip(x_3)
    x_3 = (x_3[1:]-x_3[:-1])/x_3[:-1]
    
    years = x_4_1 #save it here to get years later, because we overwrite it otherwise.
    x_4_1 = np.array([x[6] for x in x_4_1[1:] if ((x[0]=='USA') and (float(x[5][:4])>=y_min and float(x[5][:4])<=y_max))]).astype(np.float64)
    x_4_1 = x_4_1[:-1]
    
    x_4_2 = np.array([x[6] for x in x_4_2[1:] if (float(x[5][:4])>=y_min and float(x[5][:4])<=y_max)]).astype(np.float64)
    x_4_2 = x_4_2[:-1]
    x_4 = x_4_2 - x_4_1
     
    Y = np.array([x[2]  for x in y[1:] 
                   if x[0][5:7] in {'3-','6-', '9-', '12'} ]).astype(np.float64)
    
    years = np.array([x[5] for x in years[1:] if ((x[0]=='USA') and (float(x[5][:4])>=y_min and float(x[5][:4])<=y_max))])
    years = years[:-1]
    X = np.array([x_1,x_2, x_3\
                  ,x_4]).transpose()
    return X,Y.reshape(-1,1),years

def plotTable(data, rowList = None,colList = None, colLabel = None, rowLabel = None, type = None, MacroFlag = False, trueM = None):
    """ 
    Plots input matrices as tables
    
    Parameters
    ----------
    type: str
        Can be 'meanM', 'meanMWhite', 'meanR' and 'std', depending on whether one wants to 
        plot the average trunc. order while knowing the true order, 
        plot the average trunc. order whithout knowing the true order,
        plot the average R^2 value or
        plot any type of std.
    
    MacroFlag: boolean
        Indicates whether the tables contains synthetically data or macro data.
        This changes some specificationy, e.g. the labels of the axis.
    
    trueM: int
        Indicates the true truncation order if it is known. Determines coloring 
        of the table containing average trucnation order.
    """
    
    if MacroFlag == True:
        fig = plt.figure(figsize=(9,2.5))
    else:
        fig = plt.figure(figsize=(7,2.5))
        
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    
    data = np.around(data,2)
    
    if type == 'meanM':
        
        norm = .8*np.ones(data.shape)
        if trueM != None:
            if np.max(data) != np.min(data):
                norm = data/(np.max(data)-np.min(data))
            
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = [0,.5,.6,.8,0.9,.95,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
        colours = cmap(norm)
    elif type == 'meanMwhite':
        norm = 0.8*np.ones(shape=data.shape) #data/(np.max(data)-np.min(data))
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = [0,.5,.6,.8,0.9,.95,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
        colours = cmap(norm)
    elif type == 'meanR':
        norm = data +0.5 #(data +3 )/(np.max(data)+3)
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v= [0.,0.1,0.2,0.5,0.6,0.7,0.8]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        colours = cmap(norm)
        
    elif type == 'std':
        norm = data/2
        c = ["darkgreen","green","palegreen","white","lightcoral","red","darkred"]
        v = [0,.1,.2,.4,0.6,.8,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
        colours = cmap(norm)
        
    data = data.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] =  "{:5.2f}".format(data[i][j])              
        
    the_table = ax.table(cellText=data, colLabels=colList, rowLabels = rowList, loc='center',
             cellColours=colours, colWidths = np.ones((len(colList)))/(len(colList)+6))
    
    the_table.auto_set_font_size(False)
    
    if MacroFlag == True:
        the_table.set_fontsize(10)
        ax.set_ylabel(rowLabel, fontsize=13)
        ax.set_xlabel(colLabel, fontsize=13)

    else:
        the_table.set_fontsize(13)
        ax.set_ylabel(rowLabel, fontsize=15)
        ax.set_xlabel(colLabel, fontsize=15)
        
    the_table.scale(1.3, 1.5)


    ax.xaxis.set_label_position('top')

    plt.show()
    
    return