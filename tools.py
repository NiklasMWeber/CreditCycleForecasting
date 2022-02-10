import csv
import numpy as np
#from statistics import mean
#from itertools import groupby
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
    x = []
    with open(fileWithPath, 'r') as file:
        file = csv.reader(file, delimiter=delimiter)
        for row in file:
            x.append(row)
    return x

# def importData():
#     ''' 
#     Maybe paths as input?
#     '''
#     gdpPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\gdp_growth_north_america.csv'
#     unemplPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\unemployment_north_america.csv'
#     indexPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\s&p500.txt'
#     stirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\short_term_us.csv'
#     ltirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\long_term_us.csv'
#     pdPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\PD_North America.csv'
    
#     x_1 = importFile(gdpPath, delimiter = ';')
#     x_2 = importFile(unemplPath, delimiter = ';')
#     x_3 = importFile(indexPath, delimiter = '\t')
#     x_4_1 = importFile(stirPath, delimiter = ',')
#     x_4_2 = importFile(ltirPath, delimiter = ',')
#     y = importFile(pdPath, delimiter = ',')
    
#     return x_1,x_2,x_3,x_4_1,x_4_2,y

def importDataQ():
    ''' 
    Maybe paths as input?
    '''
    #C:\Users\nikth\Documents\Uni\Dateien\Semester 11\MA\Data
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

# def prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y):
#     ''' 
#     Maybe years as input?
#     '''
#     y_min = 1991
#     y_max = 2020
    
#     #just filter for right years
#     x_1 = np.array([x for x in x_1 if (float(x[0])>=y_min and float(x[0])<=y_max)]).astype(np.float64)
#     #just filter for right years
#     x_2 = np.array([x for x in x_2 if (float(x[0])>=y_min and float(x[0])<=y_max)]).astype(np.float64)
#     #extract dates -> average years -> take right years
#     x_3 = [[x[0].split()[2],float(x[1].replace(',',''))] for x in x_3] #now its a list with [[year,value] possibly with several entries for every year (months,days...)]
#     x_3 = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(x_3, lambda x: x[0]) if (y_min-1 <= float(key) and float(key) <=y_max)])
#     x_3 = np.flip(x_3,0).astype(np.float64)
#     x_3[1:,1] = ((x_3[range(1,len(x_3))]-x_3[range(0,len(x_3)-1)])/x_3[range(0,len(x_3)-1)])[:,1]
#     x_3 = x_3[1:,:]
    
#     x_3_squared = x_3.copy()
#     x_3_squared[:,1] = x_3[:,1]**2
    
#     # extract date and val -> find intersection for dates -> calc diff -> average year-wise
#     x_4_1 = [[x[-3],x[-2]] for x in x_4_1[1:]]
#     x_4_2 = [[x[-3],x[-2]] for x in x_4_2[1:]]
    
#     dates_1 = [x[0] for x in x_4_1]
#     dates_2 = [x[0] for x in x_4_2]
#     x_4_1 = [[x[0],x[1]] for x in x_4_1 if x[0] in dates_2]
#     x_4_2 = [[x[0],x[1]] for x in x_4_2 if x[0] in dates_1]
    
#     x_4 = [[x[0], float(y[1])-float(x[1])] for x,y in zip(x_4_1,x_4_2)]
#     x_4 = [[x[0].split('-')[0],x[1]] for x in x_4]
#     x_4 = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(x_4, lambda x: x[0]) if (y_min <= float(key) and float(key) <=y_max)])
#     x_4 = x_4.astype(np.float64)
    
#     y = [[y[0].split('-')[0],float(y[4])]for y in y[1:]]
#     y = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(y, lambda x: x[0]) if (y_min <= float(key) and float(key) <=y_max)])
    
#     Y= y[:,1].astype(np.float64).reshape(-1, 1)
#     years = y[:,0].astype(np.int64).reshape(-1, 1)
#     X = np.array([x_1[:,1],x_2[:,1], x_3[:,1]\
#                   #,x_3_squared[:,1]\
#                       ,x_4[:,1]],dtype=object).transpose()
#     return X,Y,years

def prepareDataQ(x_1,x_2,x_3,x_4_1,x_4_2,y):
    ''' 
    Maybe years as input?
    '''
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
                  #,x_3**2\
                      ,x_4]).transpose()
    return X,Y.reshape(-1,1),years


def plotTable(data, rowList = None,colList = None, colLabel = None, rowLabel = None, type = None, Qflag = False, trueM = None):

    if Qflag == True:
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
        #redVal = 3
        #greenVal = 0.5
        norm = data +0.5 #(data +3 )/(np.max(data)+3)
        #zero = 3/(np.max(data)+3)
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v= [0.,0.1,0.2,0.5,0.6,0.7,0.8]
        #[0.,zero/3,2*zero/3,zero,zero+(1-zero)/3 ,zero+2*(1-zero)/3 ,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        colours = cmap(norm)
        
    elif type == 'std':
        norm = data/2
        #c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
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
    
    if Qflag == True:
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
    

# X,Y,years = prepareData(importData()[0],importData()[1],importData()[2],importData()[3],importData()[4],importData()[5])
# mat = np.concatenate((X,Y), axis = 1)
# mat = np.concatenate((years,mat), axis = 1)

#mat = np.load('macrodata.npy')
#X,Y,years = mat[:,1:-2], mat[:,-1].reshape((-1,1)), mat[:,0].reshape((-1,1))


if __name__ == '__main__':
    x_1,x_2,x_3,x_4_1,x_4_2,y = importDataQ()
    X,Y,years = prepareDataQ(x_1,x_2,x_3,x_4_1,x_4_2,y)
    mat = np.concatenate((years.reshape(-1,1), np.concatenate((X,Y), axis = 1)),axis = 1)
    # del x_1,x_2,x_3,x_4_1,x_4_2,y
    #a=5
    #plotTable(data = mHat_Matrix[:,:,0], rowList = nPathsList,colList = numForPartitionList, colName = 'nPoints', rowName = 'nPaths')
