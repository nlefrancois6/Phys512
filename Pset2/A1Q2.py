#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:33:00 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
import math


#polynomial fit for comparison
def poly_fit(x, y, n):
    """
    x, y: data points
    n: order of the polynomial fit
    
    params: parameters for the legendre polynomial fit
    fit: values of the legendre polynomial fit at each x point
    """
    params = np.polynomial.legendre.legfit(x, y, n)
    fit = np.polynomial.legendre.legval(x, params)
    
    return params, fit

#Set up the chebyshev polynomial matrix
def cheb_mat(x, n):
    """
    x: x-data
    n: order of the polynomial fit
    
    A: matrix where each column is a chebyshev polynomial using the recurrence
    relation T_{n+1} = 2xT_{n}-T_{n-1}, T_{0} = 1, T_{1} = x
    """
    A = np.zeros([len(x), n])
    #T_{0} = 1
    A[:,0] = 1
    #T_{1} = x
    if n>1:
        A[:,1] = x
    #T_{n+1} = 2xT_{n}-T_{n-1}
    for i in range(2, n):
        A[:,i] = 2*x*A[:,i-1] - A[:,i-2]
        
    return A

#Calculate optimal chebyshev polynomial coefficients using the least squares fit
def cheb_fit(x, y, n, A):
    """
    x, y: data points
    n: order of the polynomial fit
    
    params: parameters for the chebyshev polynomial fit
    fit: values of the chebyshev polynomial fit at each x point
    """
    
    d = np.matrix(y).transpose()
    [u, s, vt] = np.linalg.svd(A, full_matrices = False)
    sinv = np.matrix(np.diag(1.0/s))
    params = vt.transpose()*sinv*(u.transpose()*y)
    
    return params

#Evaluate the chebyshev polynomial that we have fit, at each x-data point
def cheb_eval(A, params):
    y_matrix = np.dot(A, params)
    
    y = np.squeeze(np.asarray(y_matrix.sum(axis=1)))
    return y
    

#Calculate the maximum error of the fit
def fit_error(y, fit):
    err = np.abs(y-fit)
    err_max = max(err)
    
    return err_max

#Calculate the root mean square error of the fit
def rms_error(y, fit):
    mse = ((y-fit)**2).mean()
    
    return np.sqrt(mse)

#Find the minimum order n needed to acheive a fit error satisfying the tolerance
def find_min_order(x, y, tol):
    #Initialize n and err
    n = 1
    err = tol + 1
    
    while err > tol:
        #Change this to chebyshev fit once it works
        """
        [params, fit] = poly_fit(xp, yp, n)
        err = fit_error(xp, yp, fit)
        """
        A = cheb_mat(xp, n)
        params = cheb_fit(xp, yp, n, A)
        y_fit = cheb_eval(A, params)
        
        err = fit_error(yp, y_fit)      
        n = n+1
    
    print('Accuracy satisfies tolerance for order ', n, ' polynomial')
    return n, err

#Define the function y = log_{2}(x) over a set of x-data points
NP = 100
xp = np.linspace(0.5,1,NP)
yp = np.log2(xp)

#Find the number of terms needed to meet the tolerance with the chebyshev fit
tol = 1e-6
n_tol, cheb_err_max = find_min_order(xp, yp, tol)

#Evaluate the cheb fit and check the error
n_cheb = n_tol
A_cheb = cheb_mat(xp, n_cheb)
params_cheb = cheb_fit(xp, yp, n_cheb, A_cheb)
y_cheb = cheb_eval(A_cheb, params_cheb)
cheb_err_rms = rms_error(yp, y_cheb)
print('Maximum error of Chebyshev fit: ', cheb_err_max)
print('RMS error of polynomial fit: ', cheb_err_rms)

#Calculate residuals
cheb_res = yp - y_cheb

#Evaluate the poly fit and check the error
n_poly = n_tol
[params_poly, y_poly] = poly_fit(xp, yp, n_poly)
poly_err_max = fit_error(yp, y_poly)
poly_err_rms = rms_error(yp, y_poly)
print('Maximum error of polynomial fit: ', poly_err_max)
print('RMS error of polynomial fit: ', poly_err_rms)

"""
The chebyshev fit has a higher error than the polynomial error when considering 
both the maximum error and the RMS error, by about 1 order of magnitude in each case
"""

#Calculate residuals
poly_res = yp - y_poly


#Plot the results
fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1,0.3]})
ax1.plot(xp, yp, '.k', label = 'Data')
ax1.plot(xp, y_poly, 'b', label = 'Polyfit')
ax1.legend()
ax1.set_title('Polynomial Fit')
ax1.set_ylabel('f(x)')

ax2.plot(xp, poly_res, 'b')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('x')


fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1,0.3]})
ax1.plot(xp, yp, '.k', label = 'Data')
ax1.plot(xp, y_cheb, 'r', label = 'Chebyshev Fit')
ax1.legend()
ax1.set_title('Chebyshev Fit')
ax1.set_ylabel('f(x)')

ax2.plot(xp, cheb_res, 'r')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('x')




