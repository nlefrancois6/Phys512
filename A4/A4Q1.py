#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:46:31 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

#Define the gaussian function
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#Shift an array f by an arbitrary amount shift by convolving with a delta function
def shiftFunc(x, f, shift):
    k=np.arange(len(f))
    #Convolution
    F = np.fft.fft(f)
    D = np.exp(-2j*np.pi*k*shift/len(f))
    fShifted = np.fft.ifft(F*D)
    return fShifted

#Define a gaussian and an amount of indices to shift it by
Np = 100
xp = np.linspace(-np.pi,np.pi,Np)
#xp = np.linspace(-10,10,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
shift = Np/2
#Calculate the shifted gaussian array
h = shiftFunc(xp, f, shift)

#Plot the results
plt.plot(xp, gaussian(xp,mu,sig),'b',label='Original Gaussian')
plt.plot(xp, h,'orange',label='Shifted Gaussian')
plt.legend(loc='lower right')


