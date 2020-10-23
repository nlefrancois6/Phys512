#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:51:05 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#Define the convolution operation
def conv_basic(f, g):
    F = np.fft.fft(f)
    G = np.fft.fft(g)
    return np.fft.ifft(F*G)/len(f)
#Improved convolution operation with padding to avoid wraparound issues
def conv_padded(f, g):
    #Add padding
    padLen = len(f)
    f = np.pad(f,[0,padLen],mode='constant')
    g = np.pad(g,[0,padLen],mode='constant')
    #Take convolution of padded arrays, then remove padding and take real part
    conv = conv_basic(f,g)[:-padLen].real
    return conv

#Define a gaussian and an amount to shift it by
Np = 100
xp = np.linspace(0,10*np.pi,Np)
x_extended = np.linspace(0,10*np.pi+0.5,Np)
f = np.sin(xp)
g = np.sin(x_extended)
#g = x_extended*0.1

#Plot the results
plt.plot(xp, f,label='Sine 1')
plt.plot(x_extended, g,':',label='Sine 2')
plt.plot(xp,conv_basic(f,g),label='Basic Convolution')
plt.plot(xp,conv_padded(f,g),'c',label='Improved Convolution')
plt.legend()