# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:22:08 2021

@author: Niklas Weber
"""

import tools
import numpy as np
from esig import tosig as ts
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#=============================================================================
#Test computing signature and logsig:

# stream = np.array([[1.0,1.0], [3.0,4.0],[5.0,2.0],[8.0,6.0]])
                    
# depth = 3

# sig = ts.stream2sig(stream,depth)
# logsig = ts.stream2logsig(stream,depth)
# print(ts.sigdim(2,3))
# print(sig) 
# print("\n",logsig)
#=============================================================================

# Import Data
x_1,x_2,x_3,x_4_1,x_4_2,y = tools.importData()
X,Y,year = tools.prepareData(x_1,x_2,x_3,x_4_1,x_4_2,y)
del x_1,x_2,x_3,x_4_1,x_4_2,y

# Standardize Data
# X_scaled = StandardScaler().fit_transform(X) #-mean() --> /std
max_abs_scaler = MaxAbsScaler()
Y_scaled = max_abs_scaler.fit_transform(Y-np.mean(Y))# -mean --> range [-1,1]

# Construct 3 year rolling windows:
reg_data = []
predictors = []
predictors_for_Signature = []
for i in range(3,len(year)):
    predictors.append(X[(i-3):i,:].reshape(-1))  
    predictors_for_Signature.append(X[(i-3):i,:])
    
reg_data = [np.array(predictors),Y_scaled[3:len(year)]]
    

# Split in train and test
#randomNumbers = np.random.randn(26,15)
X_train, X_test, Y_train, Y_test = train_test_split( reg_data[0]
                                                    ,reg_data[1], test_size = 0.2)

# Train and fit ridge regression
ridge_reg = Ridge(alpha = 10, fit_intercept=False)
ridge_reg.fit(X_train,Y_train)

predict_test = ridge_reg.predict(X_test)
predict_train = ridge_reg.predict(X_train)

# Plot
fig = plt.figure()
fig1 = fig.add_subplot(2,1,1)
fig.tight_layout(pad = 3.0)
xAxes = range(1,np.shape(predict_test)[0]+1)
fig1.plot(xAxes, Y_test, label = 'true')
fig1.plot(xAxes, predict_test, label = 'predict')
fig1.legend()
fig1.set_title('Testing Sample')
print('R on Test Sample = ', ridge_reg.score(X_test, Y_test))

fig2 = fig.add_subplot(2,1,2)
xAxes = range(1,np.shape(predict_train)[0]+1)
fig2.plot(xAxes, Y_train, label = 'true')
fig2.plot(xAxes, predict_train, label = 'predict')
fig2.legend()
fig2.set_title('Training Sample')
print('R on Training Sample = ', ridge_reg.score(X_train, Y_train))