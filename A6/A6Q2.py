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
sampleDist = 'Power'

N = 10**6

if sampleDist == 'Gaussian':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    mu = 0
    sig = 0.3
    g = gauss(x,mu,sig)
    a = 2
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)

    xg = np.random.normal(mu, sig, N)
    xu = 2*np.random.rand(N) - 1

    gx = gauss(xg,mu,sig)
    ex = exp(xu,a)
    critx = ex/(c*gx)
    crit_met_x = np.abs(xu) < critx

    yg = np.random.normal(mu, sig, N)
    yu = 2*np.random.rand(N) - 1

    gy = gauss(yg,mu,sig)
    ey = exp(yu,a)
    crity = ey/(c*gy)
    crit_met_y = np.abs(yu) < crity

elif sampleDist == 'Lorentzian':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    x0 = 0
    gamma = 1
    g = lorentz(x,x0,gamma)
    g = g/max(g)
    a = 0.05
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)

    xg = np.random.standard_cauchy(N)
    xu = 2*np.random.rand(N) - 1

    gx = lorentz(xg,x0,gamma)
    ex = exp(xu,a)
    critx = ex/(c*gx)
    crit_met_x = np.abs(xu) < critx

    yg = np.random.standard_cauchy(N)
    yu = 2*np.random.rand(N) - 1

    gy = lorentz(yg,x0,gamma)
    ey = exp(yu,a)
    crity = ey/(c*gy)
    crit_met_y = np.abs(yu) < crity
    
elif sampleDist == 'Power':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    p = 4
    g = power(x,p)
    g = g/max(g)
    a = 2
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)

    #Power Law Dist
    xL = np.random.power(p, int(N/2)) - 1
    xR = 1 - np.random.power(p, int(N/2))
    xg = np.append(xL, xR)
    np.random.shuffle(xg)
    
    xu = 2*np.random.rand(N) - 1

    gx = power(xg,p)
    ex = exp(xu,a)
    critx = ex/(c*gx)
    crit_met_x = np.abs(xu) < critx

    yL = np.random.power(p, int(N/2)) - 1
    yR = 1 - np.random.power(p, int(N/2))
    yg = np.append(yL, yR)
    #np.random.shuffle(yg)
    
    yu = 2*np.random.rand(N) - 1

    gy = power(yg,p)
    ey = exp(yu,a)
    crity = ey/(c*gy)
    crit_met_y = np.abs(yu) < crity


xe = []
ye = []
for i in range(N):
    if crit_met_x[i]:
        xe.append(xu[i])
    if crit_met_y[i]:
        ye.append(yu[i])

numVars = min([len(xe),len(ye)])
efficiency = numVars/N
print('Efficiency: ', efficiency)


plt.figure()
plt.subplot(131)
plt.plot(xe[:numVars],ye[:numVars],',')
plt.title('Exponential from ' + sampleDist)

plt.subplot(132)
plt.hist(xe,density=True)
plt.subplot(133)
plt.hist(ye,density=True)




