# -*- coding: utf-8 -*-
"""
@author: Prashanth
"""

#Importing libraries
import numpy as np
import pandas as pd
import heapq
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pickle
#Importing Dataset
Sensor_1_2_comp = pd.read_csv('C:/Users/Admin/Downloads/C95F3C90B7E_30_7_2021.csv')

inp_data = Sensor_1_2_comp.dropna()
inp_data.tail(2)

inp_data.columns

y_train = inp_data['Avg_from_Actual_1.25'].values
X_train = inp_data['Actual_RPM'].values
X_test = inp_data['Avg_from_Prv_1.25'].values

X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

#Save the Model to file in the current working directory
Pkl_Filename = "Pickle_Ply_Model.pkl"
with open (Pkl_Filename, 'wb') as file:
    pickle.dump(lin_reg, file)
    
# #Load the Model back from file
# with open (Pkl_Filename, 'rb') as file:
#     Pickle_Ply_Model = pickle.load(file)
    
# y_pred = Pickle_Ply_Model.predict(poly_reg.fit_transform(X_test))

# # y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))
# Predicted = pd.DataFrame(y_pred, columns = ['Predicted'])
# Predicted

# lin_reg.predict(poly_reg.fit_transform([[1050]]))

# import os
# os. getcwd()
