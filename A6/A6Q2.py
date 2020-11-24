#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:19:54 2020

@author: noahlefrancois
"""

import numpy as np
from matplotlib import pyplot as plt

def gauss(x,mu,sig):
    return np.exp(-(x-mu)**2/(2*sig**2))
  
def exp(x,a):
    return a*np.exp(-a*np.abs(x))

def lorentz(x, x0, gamma):
    return 1/(np.pi*gamma*(1 + ((x-x0)/gamma)**2))

def power(x, a):
    y = np.empty(len(x))
    for i in range(len(x)):
        if x[i]<0:
            y[i] = a*x[i]**(a-1) - 1
        else:
            y[i] =  a*x[i]**(a-1) - 1
    return y


#options 'Gaussian','Lorentzian','Power'
sampleDist = 'Gaussian'

N = 10**6

if sampleDist == 'Gaussian':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    mu = 0
    sig = 0.4
    g = gauss(x,mu,sig)
    a = 3
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)

    xg = np.random.normal(mu, sig, N)
    xu = 2*np.random.rand(N) - 1
    yg = np.random.normal(mu, sig, N)
    yu = 2*np.random.rand(N) - 1
    
    rg = np.sqrt(xg**2+yg**2)
    ru = np.sqrt(xu**2+yu**2)
    
    gr = gauss(rg,mu,sig)
    er = exp(ru,a)
    crit = er/(c*gr)
    crit_met = np.abs(ru) < crit
    
    num_uniform_used = 4*N
    

elif sampleDist == 'Lorentzian':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    x0 = 0
    gamma = 2
    g = lorentz(x,x0,gamma)
    g = g/max(g)
    a = 0.01
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)
    
    xl = np.random.standard_cauchy(N)
    xu = 2*np.random.rand(N) - 1
    yl = np.random.standard_cauchy(N)
    yu = 2*np.random.rand(N) - 1
    
    rl = np.sqrt(xl**2+yl**2)
    ru = np.sqrt(xu**2+yu**2)
    
    lr = lorentz(rl,x0,gamma)
    er = exp(ru,a)
    crit = er/(c*lr)
    crit_met = np.abs(ru) < crit
    
    num_uniform_used = 4*N
    
elif sampleDist == 'Power':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    p = 2
    g = power(x,p)
    g = g/max(g)
    a = 3
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)

    #Power Law Dist
    xL = np.random.power(p, int(N/2)) - 1
    xR = 1 - np.random.power(p, int(N/2))
    xp = np.append(xL, xR)
    np.random.shuffle(xp)
    xu = 2*np.random.rand(N) - 1
    
    yL = np.random.power(p, int(N/2)) - 1
    yR = 1 - np.random.power(p, int(N/2))
    yp = np.append(yL, yR)
    yu = 2*np.random.rand(N) - 1
    
    rp = np.sqrt(xp**2+yp**2)
    ru = np.sqrt(xu**2+yu**2)
    
    pr = power(rp,p)
    er = exp(ru,a)
    crit = er/(c*pr)
    crit_met = np.abs(ru) < crit
    
    num_uniform_used = 4*N


#Drop the rejected values
xe = xu[xu*crit_met != 0] 
ye = yu[yu*crit_met != 0]
#Normalize to [0,1]
xe = xe/max(xe) 
ye = ye/max(ye)


numVars = len(xe) #xe=ye always, now that we're accepting based on r not x&y independently
efficiency = 2*numVars/num_uniform_used #Should actually replace N with num_uniform_used
print('Efficiency: ', efficiency)


plt.figure()
plt.subplot(131)
plt.plot(xe,ye,',') #These should be the same now that we're accepting based on r not x&y
plt.title('Exponential from ' + sampleDist)

plt.subplot(132)
plt.hist(xe,density=True)
plt.subplot(133)
plt.hist(ye,density=True)




