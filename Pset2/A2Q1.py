#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:32:18 2020

@author: noahlefrancois
"""

import numpy as np

def simpson(f, a, b, n=2):
    """
    Integrate f from a to b using Simpson's rule by splitting into two rectangles
    """
    #Handle the interval cases
    if a == b:
        return 0
    if a < b:
        sign = 1
    else:
        sign = -1
    h = (a+b)/2
    fh = f(h)
    dx = (b-a)/(3*n)
    I = sign*dx*(f(a)+ 4*fh+ f(b))
    
    return I, h, fh

def simpsonCount(f, a, b, xStore, yStore, count, n=2):
    #Handle the interval cases
    if a == b:
        return 0
    if a < b:
        sign = 1
    else:
        sign = -1
    h = (a+b)/2
    x = [a, b, h]
    y = [0, 0, 0]
    for i in range(len(x)):
        if np.any(x[i]==np.array(xStore)):
            y[i] = yStore[xStore.index(x[i])]
        else:
            y[i] = f(x[i]);
            count = count +  1
            xStore.append(x[i]);  yStore.append(y[i])
    fh = y[2]
    dx = (b-a)/(3*n)
    I = sign*dx*(y[0]+ 4*fh+ y[1])
    
    return I, h, fh, count, xStore, yStore

def simpleIntegrator(x, f, a, b, h, guess, tol, nmax, count):
    #dx = (b-a)/(len(x)-1)
    #y = f(x)
    count = 0
    
    if count < nmax:
        [guess_ah, l_h, l_fh] = simpson(f, a, h)
        [guess_hb, r_h, r_fh] = simpson(f, h, b)
        err = np.abs(guess_ah+guess_hb - guess)
        count = count + 1
        if err < tol:
            return guess, err, count
        else:
            try:
                [guess_l, err_l, count_l] = simpleIntegrator(x, f, a, h, l_h, guess_ah, tol/2, nmax, count)
            except:
                guess_l = 0
                err_l = 0
                count_l = count + 1
            try:
                [guess_r, err_r, count_r] = simpleIntegrator(x, f, h, b, r_h, guess_hb, tol/2, nmax, count)
            except:
                guess_r = 0
                err_r = 0
                count_r = count + 1
            count = count + count_l + count_r
            guess = guess_r + guess_l
            err = err_l + err_r
         
            return guess, err, count
    else:
        print('Integral did not converge after {} steps'.format(nmax))
        return guess, 1, count

#Improved version where we check the stored values before calling f
def improvedIntegrator(x, f, a, b, h, guess, tol, nmax, count, xStore = [], yStore=[]):
    count = 0
    #(a, b, xStore, yStore, n=2, count)
    
    if count < nmax:
        [guess_ah, l_h, l_fh, count, xStore, yStore] = simpsonCount(f, a, h, xStore, yStore, count)
        [guess_hb, r_h, r_fh, count, xStore, yStore] = simpsonCount(f, h, b, xStore, yStore, count)
        err = np.abs(guess_ah+guess_hb - guess)
        if err < tol:
            return guess, err, count
        else:
            try:
                [guess_l, err_l, count_l] = improvedIntegrator(x, f, a, h, l_h, guess_ah, tol/2, nmax, count, xStore, yStore)
            except:
                guess_l = 0
                err_l = 0
                count_l = count + 1
            try:
                [guess_r, err_r, count_r] = improvedIntegrator(x, f, h, b, r_h, guess_hb, tol/2, nmax, count, xStore, yStore)
            except:
                guess_r = 0
                err_r = 0
                count_r = count + 1
                
            count = count + count_l + count_r
            guess = guess_r + guess_l
            err = err_l + err_r
         
            return guess, err, count
    else:
        print('Integral did not converge after {} steps'.format(nmax))
        return guess, 1, count

#Test the integrators by evaluating a few functions
def polyf(x):
    return 3*x**3 + x**2 + 4*x - 6
def lorentz(x):
    return 1/(1-x**2)
def gaussian(x):
    return np.exp(-0.5*x**2)

xp = np.linspace(-1, 1, 10)

#Take the first step using simpson
[guess, h, fh] = simpson(polyf, -1, 1)
#Evaluate using the simple integrator
I_simple, err_simple, count_simple = simpleIntegrator(xp, polyf, -1, 1, h, guess, 1e-7, 100, 1)
#Evaluate using the improved integrator
I_imp, err_imp, count_imp = improvedIntegrator(xp, polyf, -1, 1, h, guess, 1e-7, 100, 1)



