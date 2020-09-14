#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:32:27 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

#Load the data and flip the arrays to list data in order of increasing voltage
#temp, voltage, d = np.loadtxt("lakeshore.csv",delimiter=',',unpack=True)
df = pd.read_csv("lakeshore.csv")
df = df.iloc[::-1]

#Rescale the derivative by 0.001 to match the units
df['Deriv'] = 0.001*df['Deriv']

#Create a cubic spline to represent the data
#s=0 prevents smoothing during the fit
tck = interpolate.splrep(df['Volt'], df['Temp'], s=0)


#Initialize array of voltages
vp = np.linspace(df['Volt'][0],df['Volt'][len(df)-1],1000)

#Evaluate the interpolant defined by the spline tck at a specified voltage
def interpolant1(voltage, tck):
    temp = interpolate.splev(voltage, tck, der=0)
    return temp

Tp = interpolant1(vp, tck)

#Calculate the error by taking the average of the errors at the 3 closest data points
def error_estimate1_3ptsStencil(voltage, tck, df):
    #Find nearest 3 voltages in data set
    volt_distances = np.abs(df['Volt']-voltage)
    v_closest = min(volt_distances)
    
    #Get the 3 closest data points
    i_closest = volt_distances[volt_distances==v_closest].index.values
    i_upper = i_closest + 1
    i_lower = i_closest - 1
    
    closest = df.iloc[i_closest]
    upper = df.iloc[i_upper]
    lower = df.iloc[i_lower]
    
    #Compute the error at each point and take the average
    V1err = np.abs(closest.iloc[0]['Temp'] - interpolant1(closest.iloc[0]['Volt'], tck))
    V2err = np.abs(upper.iloc[0]['Temp'] - interpolant1(upper.iloc[0]['Volt'], tck))
    V3err = np.abs(lower.iloc[0]['Temp'] - interpolant1(lower.iloc[0]['Volt'], tck))
    
    errAvg = (V1err + V2err + V3err)/3
    
    return errAvg

err = error_estimate1_3ptsStencil(0.9, tck, df)
print('Interpolation Error: ', err)
    
plt.plot(df['Volt'], df['Temp'], marker='.', markersize = 4, linestyle = '', color = 'red', label = 'Data')
plt.plot(vp, Tp, '--', color = 'black', label = 'Interpolant')
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Response of Lakeshore DT-670 Diode')
    
    
