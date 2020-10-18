#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:10:58 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def correlation(f,g):
    a = np.fft.fft(f)*np.conj(np.fft.fft(g))
    return np.fft.ifft(a)

#Define the two gaussians we want to take the correlation function of
Np = 100
xp = np.linspace(-5,5,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
g = gaussian(xp,mu,sig)

#Plot the results
plt.plot(xp, f)
plt.plot(xp, correlation(f,g))