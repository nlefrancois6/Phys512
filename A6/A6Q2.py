#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:19:54 2020

@author: noahlefrancois
"""

"""
See Q2_gaussianGeneratorOutput.png, Q2_lorentzianGeneratorOutput.png, 
and Q2_powerGeneratorOutput.png for the results plots discussed here.

Despite my expectation that the Gaussian generator would produce the best results,
I actually had a very difficult time finding a pair of parameters sig & a that gave
a good exponential distribution using the Gaussian generator. I couldn't do so at all
with the 1D generator, but was able to get it to work using my 2D generator which applies
the acceptance criterion to the radius instead of x and y separately. However, despite this
difficulty/sensitivity in the parameters, the efficiency of this generator was slightly
higher than the other two (even after accounting for using twice as many random variates).

The Lorentzian generator gave better results for varied distribution parameters than
the Gaussian generator and better efficiency than the power law.

Surprisingly, the Power Law generator actually gives significantly better results than the
other two generators even though the power law distribution looks very different from
the exponential distribution for the  parameters used. The drawback is that this
generator also has the lowest efficiency (but still within an order of magnitude of the
highest efficiency).

Gaussian Efficiency:  0.07342525
Lorentzian Efficiency: 0.0324515
Power Law Efficiency: 0.0129155

I tried to tweak the parameters used for each distribution to maximize efficiency, but
the accuracy of the distribution suffered pretty rapidly. I also replaced all the loops
with vectorized operations which reduced my execution time by a factor of ~10.
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
    sig = 0.4
    g = gauss(x,mu,sig)
    a = 2
    e = exp(x,a)
    e = e/max(e)
    c = max(e/g)

    plt.plot(x,g)
    plt.plot(x,e)
 
    xg = np.random.normal(mu, sig, N)
    xu = 2*np.random.rand(N) - 1
    yg = np.random.normal(mu, sig, N)
    yu = 2*np.random.rand(N) - 1
    
    rg = np.sqrt(xg**2+yg**2) #Get the radial distance of the point (xg, yg)
    ru = np.sqrt(xu**2+yu**2)
    
    gr = gauss(rg,mu,sig)
    er = exp(ru,a)
    #Check the criterion against the radial position instead of the x and y positions separately
    crit = er/(c*gr)
    crit_met = np.abs(ru) < crit
    
    num_uniform_used = 4*N

elif sampleDist == 'Lorentzian':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    x0 = 0
    gamma = 1.0#0.5, 1.0
    g = lorentz(x,x0,gamma)
    g = g/max(g)
    a = 1.4 #0.7, 1.4
    e = exp(x,a)
    e = e/max(e)
    c = max(g/e)

    plt.plot(x,g)
    plt.plot(x,e)
    
    xl = np.random.standard_cauchy(N)
    xu = 2*np.random.rand(N) - 1
    
    lx = lorentz(xl,x0,gamma)
    ex = exp(xu,a)
    crit = lx/(c*ex)
    crit_met = np.abs(xu) < crit
    
    num_uniform_used = 2*N
    
elif sampleDist == 'Power':
    #Calculate c s.t. e/g <= c
    x = np.linspace(-1,1,1000)
    p = 2
    g = power(x,p)
    g = g/max(g)
    a = 4
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
    
    px = power(xp,p)
    ex = exp(xu,a)
    crit = ex/(c*px)
    crit_met = np.abs(xu) < crit
    
    num_uniform_used = 2*N


#Drop the rejected values
xe = xu[xu*crit_met != 0] 
#Normalize to [0,1]
xe = xe/max(xe) 

numVars = len(xe)
efficiency = numVars/num_uniform_used 
print('Efficiency: ', efficiency)


plt.figure()
plt.subplot(121)
plt.plot(xe,',')
plt.title('Exponential from ' + sampleDist)

plt.subplot(122)
n, b, p = plt.hist(xe,density=True)
plt.plot(x,e*max(n), label='Exponential')
plt.plot(x,g*max(n), label=sampleDist)
plt.legend()

