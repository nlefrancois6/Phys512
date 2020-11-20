#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:26:49 2020

@author: noahlefrancois
"""

import numpy as np
from matplotlib import pyplot as plt

#Generate exponential var using box-muller from uniform var
N=10**5
x=2*np.random.rand(N)-1
y=2*np.random.rand(N)-1
rsqr=(x**2+y**2)
ind=rsqr<1
x=x[ind]
y=y[ind]
rsqr=rsqr[ind]
rr=np.sqrt(-2*np.log(rsqr)/rsqr)
rr=rr #normalize
xx=x*rr
yy=y*rr

xx = xx/max(xx)
yy = yy/max(yy)

print(str(len(x)/N), 'Uniform-Exp Box-Muller Efficiency')

#Gaussian
xg = np.random.normal(0, 0.3, N)
yg = np.random.normal(0, 0.3, N)

rsqr=(xg**2+yg**2)
ind=rsqr<1
xg=xg[ind]
yg=yg[ind]

#Compare Lorentzian, Gaussian, and Power vars to Exponential to assess potential bounding distributions

#Lorentzian
xl = np.random.standard_cauchy(N)
yl = np.random.standard_cauchy(N)

rsqr=(xl**2+yl**2)
ind=rsqr<10
xl=xl[ind]
yl=yl[ind]
xl = xl/max(abs(xl))
yl = yl/max(abs(yl))

#Power Law
a=4
xL = np.random.power(a, int(N/2)) - 1
xR = 1 - np.random.power(a, int(N/2))
xp = np.append(xL, xR)
yL = np.random.power(a, int(N/2)) - 1
yR = 1 - np.random.power(a, int(N/2))
yp = np.append(yL, yR)

np.random.shuffle(xp) #need to shuffle since the signs of the original shifted xp & yp are correlated

plt.figure()
plt.subplot(241)
plt.plot(xx, yy,',')
plt.title('Exponential')

plt.subplot(242)
plt.plot(xg, yg,',')
plt.title('Gaussian')

plt.subplot(243)
plt.plot(xl, yl,',')
plt.title('Lorentzian')

plt.subplot(244)
plt.plot(xp, yp,',')
plt.title('Power Law')

plt.subplot(245)
plt.hist(xx, density=True)

plt.subplot(246)
plt.hist(xg, density=True)

plt.subplot(247)
plt.hist(xl, density=True)

plt.subplot(248)
plt.hist(xp, density=True)




