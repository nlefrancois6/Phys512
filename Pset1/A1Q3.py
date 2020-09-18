#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:42:23 2020

@author: noahlefrancois
"""

import numpy as np
from scipy.interpolate import CubicSpline

#Define the true function with high resolution
x_true_C = np.linspace(-np.pi/2, np.pi/2, 1000)
cos_true = np.cos(x_true_C)

x_true_L = np.linspace(-1, 1, 1000)
lorentz_true = 1/(1+x_true_L**2)

#Define the 'data' with small np
n=4
m=5
xp_C = np.linspace(-np.pi/2, np.pi/2, n+m-1)
f_C = np.cos(xp_C)

xp_L = np.linspace(-1, 1, n+m-1)
f_L = 1/(1+xp_L**2)


#Fit a polynomial function, evaluate it & get the error compared to f_true
def poly_interp(xp, f_np, n, m, x_true, f_true):
    poly = np.polyfit(xp, f_np, n+m-1)
    f_poly = np.polyval(poly, x_true)
    err_poly = sum(np.abs(f_true-f_poly))/len(f_true)
    
    return poly, err_poly

#Fit a cubic spline, evaluate it & get the error compared to f_true
def cubicSpline_interp(xp, f_np, x_true, f_true):
    cs = CubicSpline(xp, f_np)
    f_cs = cs(x_true)
    err_cs = sum(np.abs(f_true-f_cs))/len(f_true)
    
    return cs, err_cs

#Define a rational function interpolation
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

#Fit the rational function, evaluate it & get the error compared to f_true
def rat_interp(xp, f_np, n, m, x_true, f_true):
    p, q=rat_fit(xp, f_np, n, m)
    f_rat = rat_eval(p, q, x_true)
    err_rat = sum(np.abs(f_true-f_rat))/len(f_true)
    
    return p, q, err_rat

#Test the interpolation functions
[poly_C, err_poly_C] = poly_interp(xp_C, f_C, n, m, x_true_C, cos_true)
[cs_C, err_cs_C] = cubicSpline_interp(xp_C, f_C, x_true_C, cos_true)
[p_C, q_C, err_rat_C] = rat_interp(xp_C, f_C, n, m, x_true_C, cos_true)

[poly_L, err_poly_L] = poly_interp(xp_L, f_L, n, m, x_true_L, lorentz_true)
[cs_L, err_cs_L] = cubicSpline_interp(xp_L, f_L, x_true_L, lorentz_true)
[p_L, q_L, err_rat_L] = rat_interp(xp_L, f_L, 4, 5, x_true_L, lorentz_true)

"""
For n=1, m=8, the error of the lorentz rational fit is 1e-15, but for higher
order the error rapidly gets much worse: 
    (n=2, m=7, err = 0.6, n=3, m=6, err = 1.5, n=4, m=5, err = 1.2)
This makes sense because the lorentzian function is a rational function 
of order 1/x^2, so n=1 provides a good fit.

Switching from inv to pinv results in much better results at high order:
    (n=2, m=7, err = 1e-16, n=3, m=6, err = 1e-16, n=4, m=5, err = 1e-15)
as well as 1e-16 at n=1.

The lorentzian function is an even function, meaning that only even order terms
should be significant. With n=4 m=5, this was observed to be the case using 
pinv (other terms were ~1e-14), but using inv all 4 terms of both p and q were order 1
which leads to the observed large increase in error when using inv.
"""





