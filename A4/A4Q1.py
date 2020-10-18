#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:46:31 2020

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

#Define a gaussian and an amount to shift it by
Np = 100
xp = np.linspace(-5,5,Np)
mu = 0; sig = 1
f = gaussian(xp,mu,sig)
shift = 1

#Calculate the shifted function
g = shiftFunc(f, shift)

#Plot the results
plt.plot(xp, gaussian(xp,mu,sig))
plt.plot(xp, g)


