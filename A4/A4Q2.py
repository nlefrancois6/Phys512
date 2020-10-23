#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:10:58 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

#Define the gaussian function
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#Return the normalized correlation function
def correlation(f,g):
    a = np.fft.fft(f)*np.conj(np.fft.fft(g))
    corr = np.fft.ifft(a)
    return corr/max(corr)

#Define the two gaussians we want to take the correlation function of (identical)
Np = 100
xp = np.linspace(-np.pi,np.pi,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
g = gaussian(xp,mu,sig)

#Plot the results
plt.plot(xp, f, label='Gaussian')
plt.plot(xp, correlation(f,g), label='Correlation')
plt.legend()