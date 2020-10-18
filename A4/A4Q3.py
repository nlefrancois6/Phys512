#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:34:25 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#Shift an array f by an arbitrary amount shift
def shiftFunc(f, shift):
    fShifted = f
    return fShifted
def correlation(f,g):
    a = np.fft.fft(f)*np.conj(np.fft.fft(g))
    return np.fft.ifft(a)

#Define a gaussian f and an amount to shift it by
Np = 100
xp = np.linspace(-5,5,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
#Probably want to produce a plot comparing multiple shift values
shift = 1

#Calculate the shifted function g and the correlation h of f*g
g = shiftFunc(f, shift)
h = correlation(f,g)

#Plot the results
plt.plot(xp, gaussian(xp,mu,sig))
plt.plot(xp, h)