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
print('The uncertainty in parameter a is +/-', perr[3])

"""
The focal length of a parabola is F=R^2/4a, where R is the dish radius at its rim
"""
#Calculate the dimensions of the dish
h = max(zd) - min(zd)
Dx = max(xd) - min(xd)
Dy = max(yd) - min(yd)

Rx = Dx/2
F = Rx**2/(4*paramsOpt[3])
errF = np.sqrt(F**2*(perr[3]/paramsOpt[3])**2)
print('The focal length is ', F, ' +/- ', errF)

"""
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(xd, yd, zd, color = 'green')
ax.scatter3D(xd, yd, zFit, color = 'blue')
"""