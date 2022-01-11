import csv
import numpy as np
from statistics import mean
from itertools import groupby

    
def importFile(fileWithPath, delimiter):
    x = []
    with open(fileWithPath, 'r') as file:
        file = csv.reader(file, delimiter=delimiter)
        for row in file:
            x.append(row)
    return x

def importData():
    ''' 
    Maybe paths as input?
    '''
    gdpPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\gdp_growth_north_america.csv'
    unemplPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\unemployment_north_america.csv'
    indexPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\s&p500.txt'
    stirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\short_term_us.csv'
    ltirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\long_term_us.csv'
    pdPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\PD_North America.csv'
    
    x_1 = importFile(gdpPath, delimiter = ';')
    x_2 = importFile(unemplPath, delimiter = ';')
    x_3 = importFile(indexPath, delimiter = '\t')
    x_4_1 = importFile(stirPath, delimiter = ',')
    x_4_2 = importFile(ltirPath, delimiter = ',')
    y = importFile(pdPath, delimiter = ',')
    
    return x_1,x_2,x_3,x_4_1,x_4_2,y

def prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y):
    ''' 
    Maybe years as input?
    '''
    y_min = 1991
    y_max = 2019
    
    #just filter for right years
    x_1 = np.array([x for x in x_1 if (float(x[0])>=y_min and float(x[0])<=y_max)]).astype(np.float64)
    #just filter for right years
    x_2 = np.array([x for x in x_2 if (float(x[0])>=y_min and float(x[0])<=y_max)]).astype(np.float64)
    #extract dates -> average years -> take right years
    x_3 = [[x[0].split()[2],float(x[1].replace(',',''))] for x in x_3] #now its a list with [[year,value] possibly with several entries for every year (months,days...)]
    x_3 = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(x_3, lambda x: x[0]) if (y_min-1 <= float(key) and float(key) <=y_max)])
    x_3 = np.flip(x_3,0).astype(np.float64)
    x_3[1:,1] = ((x_3[range(1,len(x_3))]-x_3[range(0,len(x_3)-1)])/x_3[range(0,len(x_3)-1)])[:,1]
    x_3 = x_3[1:,:]
    
    x_3_squared = x_3.copy()
    x_3_squared[:,1] = x_3[:,1]**2
    
    # extract date and val -> find intersection for dates -> calc diff -> average year-wise
    x_4_1 = [[x[-3],x[-2]] for x in x_4_1[1:]]
    x_4_2 = [[x[-3],x[-2]] for x in x_4_2[1:]]
    
    dates_1 = [x[0] for x in x_4_1]
    dates_2 = [x[0] for x in x_4_2]
    x_4_1 = [[x[0],x[1]] for x in x_4_1 if x[0] in dates_2]
    x_4_2 = [[x[0],x[1]] for x in x_4_2 if x[0] in dates_1]
    
    x_4 = [[x[0], float(y[1])-float(x[1])] for x,y in zip(x_4_1,x_4_2)]
    x_4 = [[x[0].split('-')[0],x[1]] for x in x_4]
    x_4 = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(x_4, lambda x: x[0]) if (y_min <= float(key) and float(key) <=y_max)])
    x_4 = x_4.astype(np.float64)
    
    y = [[y[0].split('-')[0],float(y[4])]for y in y[1:]]
    y = np.array([[key, mean(map(lambda x:x[1],list(group)))]for key, group in groupby(y, lambda x: x[0]) if (y_min <= float(key) and float(key) <=y_max)])
    
    Y= y[:,1].astype(np.float64).reshape(-1, 1)
    years = y[:,0].astype(np.int64).reshape(-1, 1)
    X = np.array([x_1[:,1],x_2[:,1], x_3[:,1]\
                  ,x_3_squared[:,1]\
                      ,x_4[:,1]]).transpose()
    return X,Y,years
