#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:51:29 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit

dishData = np.loadtxt('dish_zenith.txt')
xd = dishData[:,0]; yd = dishData[:,1]; zd = dishData[:,2];
xyData = [xd, yd]

"""
The given form of the dish shape function, which is non-linear
"""
def dishShape(x, y, x0, y0, z0, a):
    z = a*((x-x0)**2+(y-y0)**2)+z0
    return z

"""
Substitute X = (x-x0)**2, Y = (y-y0)**2 in order to make the problem linear
"""
def dishShapeLinear(xyData, x0, y0, z0, a):
    #x0 = params[0]; y0 = params[1]; z0 = params[2]; a = params[3]
    x = xyData[0]; y = xyData[1];
    X = (x-x0)**2
    Y = (y-y0)**2
    z = a*(X + Y) + z0
    return z

#Fit the dish shape using Levenberg-Marquardt, get optimal parameters and covariance matrix
paramsOpt, pcov = curve_fit(dishShapeLinear, xyData, zd, method='lm')
#Calculate the function value at each data point using the optimal params
zFit = dishShapeLinear(xyData, paramsOpt[0], paramsOpt[1], paramsOpt[2], paramsOpt[3])
#Get the 1sigma errors for optimal parameters x0, y0, z0, a
perr = np.sqrt(np.diag(pcov))
#print('The uncertainty in parameter a is +/-', f'{perr[3]:.3}')
labels = ['x0','y0','z0','a']
for i in range(len(paramsOpt)):
    print(labels[i],' fit value ', f'{paramsOpt[i]:.3}','+/-', f'{perr[i]:.3}')

"""
The focal length of a parabola can be calculated as F=(1/4a)*1e-3, 
using the 1e-3 factor to convert to metres
"""
F = 1/(1000*4*paramsOpt[3])
errF = np.sqrt(F**2*(perr[3]/paramsOpt[3])**2)
print('The focal length is ', f'{F:.6}', ' +/- ', f'{errF:.3}','\n')

"""
#Plot the data points and the fit points at the same locations
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(xd, yd, zd, color = 'green')
ax.scatter3D(xd, yd, zFit, color = 'blue')
"""
"""
Output:
    x0  fit value  -1.36 +/- 0.377
    y0  fit value  58.2 +/- 0.359
    z0  fit value  -1.51e+03 +/- 0.313
    a  fit value  0.000167 +/- 6.48e-08
    The focal length is  1.49966  +/-  0.000583
    
    This fit value is within one sigma of the expected length of 1.5m.
"""

"""
BONUS QUESTION
"""
#Leading order correction for the non-symmetric dish in bonus question
def dishShapeRotation(xyData, x0, y0, z0, a, b, theta):
    x = xyData[0]; y = xyData[1];
    xrot = np.cos(theta)*x + np.sin(theta)*y
    yrot = -np.sin(theta)*x + np.cos(theta)*y
    X = (xrot-x0)**2
    Y = (yrot-y0)**2
    z = a*X + b*Y + z0
    return z

#Fit the corrected dish shape using Levenberg-Marquardt, get optimal parameters and covariance matrix
paramsOptR, pcovR = curve_fit(dishShapeRotation, xyData, zd, method='lm')
#Calculate the function value at each data point using the optimal params
zFitR = dishShapeRotation(xyData, paramsOptR[0], paramsOptR[1], paramsOptR[2], paramsOptR[3], paramsOptR[4], paramsOptR[5])
#Get the 1sigma errors for optimal parameters x0, y0, z0, a, b, theta
perrR = np.sqrt(np.diag(pcovR))
#print('The uncertainty in parameter a is +/-', f'{perr[3]:.3}')
labelsR = ['x0','y0','z0','a','b','theta']
print('Bonus Fit: General Parabola with Rotation Parameter')
for i in range(len(paramsOptR)):
    print(labelsR[i],' fit value ', f'{paramsOptR[i]:.3}','+/-', f'{perrR[i]:.3}')
    
#Output the focal length of each axis
Fa = 1/(1000*4*paramsOptR[3])
errFa = np.sqrt(Fa**2*(perrR[3]/paramsOptR[3])**2)
print('The focal length in the x direction is ', f'{Fa:.6}', ' +/- ', f'{errFa:.3}')

Fb = 1/(1000*4*paramsOptR[4])
errFb = np.sqrt(Fb**2*(perrR[4]/paramsOptR[4])**2)
print('The focal length in the y direction is ', f'{Fb:.6}', ' +/- ', f'{errFb:.3}')

"""
Output:
    Bonus Fit: General Parabola with Rotation Parameter
    x0  fit value  35.0 +/- 0.539
    y0  fit value  -46.2 +/- 0.414
    z0  fit value  -1.51e+03 +/- 0.14
    a  fit value  0.000166 +/- 3.66e-08
    b  fit value  0.000168 +/- 3.78e-08
    theta  fit value  2.52 +/- 0.0109
    The focal length in the x direction is  1.50849  +/-  0.000333
    The focal length in the y direction is  1.49022  +/-  0.000336
    
    This fit tells us that there is a non-zero rotation of (2.52+/-0.01)deg, and that
    the dish is not exactly round. There is a descrepancy of 0.0183m between the respective focal 
    lengths, which is much larger than the standard deviations which are both 0.0003.
"""