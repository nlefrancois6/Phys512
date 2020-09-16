#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:28:36 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

#define some functions to evaluate the derivative and the error

#Take the first derivative using the expression from Q1a
def my_deriv(f, x, dx):
    fp = 4*(f(x+dx)-f(x-dx))/(6*dx) - (f(x+2.0*dx)-f(x-2.0*dx))/(12*dx)
    return fp

#Exponential with param 1
def exp_1(x):
    return np.exp(1*x)

#Derivative of exponential
def expDeriv_1(x, n=5):
    return 1**n*np.exp(1*x)

#Exponential with param 0.01
def exp_p01(x):
    return np.exp(0.01*x)

#Derivative of exponential
def expDeriv_p01(x, n=5):
    return 0.01**n*np.exp(0.01*x)

#Optimal dx derived from Q1b, using machine precision e
def dxOpt(f, f5p, x, e=1e-16):
    dx = (45.0*e*f(x)/(2.0*f5p(x)))**(1/5)
    return dx

#Define location and range of dx values to try
xp = 0
dxs = np.logspace(-15,1, num=50)

#For a = 1

#Define function, the true derivative, and the predicted optimal dx
fp_true = expDeriv_1(xp, n=1)
fp = my_deriv(exp_1, xp, dxs)
dx_pred = dxOpt(exp_1, expDeriv_1, xp)

#Calculate error and best dx value
err = np.abs(fp_true - fp)
dx_best = dxs[np.argmin(err)]

#State the results and generate the convergence plot
print("Results for a = 1:")
print("Predicted Optimal dx: ", dx_pred)
print("Found Optimal dx: ", dx_best)

plt.figure()
plt.plot(dxs, err, 'k.', label='Numerical Error')
plt.axvline(dx_pred, linestyle='--', label = r"Predicted Optimal $\delta$")
plt.axvline(dx_best, linestyle='--', color='red',label = r"Found Optimal $\delta$")
plt.yscale("log")
plt.xscale("log")
plt.legend()

#For a = 0.01

#Define function, the true derivative, and the predicted optimal dx
fp_true2 = expDeriv_p01(xp, n=1)
fp2 = my_deriv(exp_p01, xp, dxs)
dx_pred2 = dxOpt(exp_p01, expDeriv_p01, xp)

#Calculate error and best dx value
err2 = np.abs(fp_true2 - fp2)
dx_best2 = dxs[np.argmin(err2)]

#State the results and generate the convergence plot
print("Results for a = 0.01:")
print("Predicted Optimal dx: ", dx_pred2)
print("Found Optimal dx: ", dx_best2)

plt.figure()
plt.plot(dxs, err2, 'k.', label='Numerical Error')
plt.axvline(dx_pred2, linestyle='--', label = r"Predicted Optimal $\delta$")
plt.axvline(dx_best2, linestyle='--', color='red',label = r"Found Optimal $\delta$")
plt.yscale("log")
plt.xscale("log")
plt.legend()



