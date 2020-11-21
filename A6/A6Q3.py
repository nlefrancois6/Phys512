#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:19:29 2020

@author: noahlefrancois
"""

import numpy as np
from matplotlib import pyplot as plt
import time

def exp(x,a):
    return a*np.exp(-a*np.abs(x))


N = 10**6
a = 2 #exponential param

t1 = time.time() #start timer

u = np.random.rand(N) 
v = np.random.rand(N)

rat = v/u 
p_rat = exp(v/u, a)

accept_crit = u <= np.sqrt(p_rat) #Check acceptance criterion
xe = rat[rat*accept_crit != 0] #Drop the rejected values
xe = xe/max(xe) #Normalize to [0,1]

t2 = time.time()
print('Run-time:',t2-t1)

efficiency = len(xe)/N
print('Generator Efficiency:', efficiency)

"""
N_in = 10**6, N_out = 423 614
Run-time: 0.2035210132598877
Generator Efficiency: 0.423614

N_in = 10**8, N_out = 423 614
Run-time: 29.858998775482178
Generator Efficiency: 0.42322922

"""

plt.figure()
plt.subplot(121)     
plt.plot(xe,',')

plt.subplot(122)
plt.hist(xe, density=True)
