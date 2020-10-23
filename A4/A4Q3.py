#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:34:25 2020

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
#Return the normalized correlation function
def correlation(f,g):
    a = np.fft.fft(f)*np.conj(np.fft.fft(g))
    corr = np.fft.ifft(a)
    return corr

#Define a gaussian f and an amount to shift it by
Np = 100
xp = np.linspace(-5,5,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
#Probably want to produce a plot comparing multiple shift values
shift = [0, Np/4, Np/2, 3*Np/4, Np]

#Calculate the shifted function g and the correlation h of f*g
fShifted1 = shiftFunc(xp, f, shift[0])
h1 = correlation(f,fShifted1)

fShifted2 = shiftFunc(xp, f, shift[1])
h2 = correlation(f,fShifted2)

fShifted3 = shiftFunc(xp, f, shift[2])
h3 = correlation(f,fShifted3)

fShifted4 = shiftFunc(xp, f, shift[3])
h4 = correlation(f,fShifted4)

fShifted5 = shiftFunc(xp, f, shift[4])
h5 = correlation(f,fShifted5)

#Plot the results
plt.plot(xp, f, label='Gaussian')
plt.plot(xp, h1, label='Shift 0')
plt.plot(xp, h2, label='Shift N/4')
plt.plot(xp, h3, label='Shift N/2')
plt.plot(xp, h4, label='Shift 3N/4')
plt.legend()

"""
See output plot A4Q3.jpg. I found that the magnitude of the correlation does not depend on the shift.
This is surprising to me since I would have guessed that the correlation would decrease in magnitude as 
the peak of the shifted Gaussian moved further away from the peak of the original Gaussian.

This behaviour is likely due to the periodic nature of the DFT, which means that even when one end of
the shifted Gaussian is moved further from the original peak (measured in the increasing x direction), 
it is also getting closer when measured in the opposite direction.
"""