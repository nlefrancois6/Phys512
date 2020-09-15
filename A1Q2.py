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
df = pd.read_csv("lakeshore.csv")
df = df.iloc[::-1]

#Rescale the derivative by 0.001 to match the units
df['Deriv'] = 0.001*df['Deriv']

temp = df['Temp'].to_numpy()
volt = df['Volt'].to_numpy()
deriv = df['Deriv'].to_numpy()

#Create a cubic spline to represent the data
#s=0 prevents smoothing during the fit
interp = interpolate.interp1d(volt, temp, kind='cubic')

#Evaluate the interpolant defined by the spline tck at a specified voltage
def f_interp(voltage, interp):
    temp = interp(voltage)
    return temp

#Estimate the error of our interpolated value
def error_estimate(voltage_in, interp, temp, volt, deriv):
    #Get the distances from each point to the voltage of interest
    volt_distances = np.abs(volt-voltage_in)
    #Pick the closest point and get its index
    ind_closest = np.where(volt_distances == np.amin(volt_distances))
    #Get the voltage, temperature, and derivative at the closest point
    v_closest = volt_distances[ind_closest[0]]
    slope_closest = deriv[ind_closest[0]]
    temp_closest = temp[ind_closest[0]]

    #Estimate the temperature by adding the distance from the point times the 
    #slope at the point to the temperature at the point
    dist = np.abs(voltage_in-v_closest)
    temp_est = temp_closest + slope_closest*dist
    
    temp_interp = f_interp(voltage_in, interp)

    err_est = np.abs(temp_est-temp_interp)
    
    return temp_interp, err_est

#Test the interpolation and error estimate functions
voltage_in = 0.8
[temp_interp, err] = error_estimate(voltage_in, interp, temp, volt, deriv)
print("Temperature at {} V is {} +/- {} K".format(voltage_in, temp_interp, err))
    
#Initialize array of voltages
vp = np.linspace(volt[0],volt[-1],1000)
#Interpolate over these points
Tp = f_interp(vp, interp)

#Plot the results for the full range of data points
plt.plot(volt, temp, marker='.', markersize = 4, linestyle = '', color = 'red', label = 'Data')
plt.plot(vp, Tp, '--', color = 'black', label = 'Interpolant')
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Response of Lakeshore DT-670 Diode')
plt.legend()