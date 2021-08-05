# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:22:08 2021

@author: Niklas Weber
"""

import numpy as np
from esig import tosig as ts
import tools

"""
Test computing signature and logsig:

stream = np.array([[1.0,1.0], [3.0,4.0],[5.0,2.0],[8.0,6.0]])
                   
depth = 3

sig = ts.stream2sig(stream,depth)
logsig = ts.stream2logsig(stream,depth)
print(sig) 
print("\n",logsig)
"""

gdpPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\gdp_growth_north_america.csv'
unemplPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\unemployment_north_america.csv'
indexPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\s&p500.txt'
stirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\short_term_us.csv'
ltirPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\long_term_us.csv'
pdPath = r'C:\Users\nikth\Dropbox\MA Gonon\Anwendung\NA_Data\PD_North America.csv'

x_1 = tools.importFile(gdpPath, delimiter = ';')
x_2 = tools.importFile(unemplPath, delimiter = ';')
x_3 = tools.importFile(indexPath, delimiter = '\t')
x_4_1 = tools.importFile(stirPath, delimiter = ',')
x_4_2 = tools.importFile(ltirPath, delimiter = ',')
y = tools.importFile(pdPath, delimiter = ',')