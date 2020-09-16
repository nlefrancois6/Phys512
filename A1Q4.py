#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:21:20 2020

@author: noahlefrancois
"""

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def simpson(f, a, b, n=2):
    """
    Integrate f from a to b using Simpson's rule by splitting into two rectangles
    """
    #Handle the interval cases
    if a == b:
        return 0
    if a < b:
        sign = 1
    else:
        sign = -1
    h = (a+b)/2
    fh = f(h)
    dx = (b-a)/(3*n)
    I = sign*dx*(f(a)+ 4*fh+ f(b))
    
    return I, h, fh

def adaptiveSimpson(f, a, b, h, guess, tol, nmax, count):
    """
    Estimate the Simpson's rule integration error and continue splitting the interval
    until the error falls below the tolerance or nmax steps have been reached
    """
    if count < nmax:
        [guess_ah, l_h, l_fh] = simpson(f, a, h)
        [guess_hb, r_h, r_fh] = simpson(f, h, b)
        err = np.abs(guess_ah+guess_hb - guess)
        if err < tol:
            return guess, count
        else:
            count = count + 1
            [guess_l, count_l] = adaptiveSimpson(f, a, h, l_h, guess_ah, tol/2, nmax, count)
            [guess_r, count_r] = adaptiveSimpson(f, h, b, r_h, guess_hb, tol/2, nmax, count)
            guess_sum = guess_l+guess_r
            count_sum = count_l+count_r
            return guess_sum, count_sum
    else:
        print('Integral did not converge after {} steps'.format(nmax))
        
def simpsonIntegrator(f, a, b, tol, nmax):
    """
    Use the Simpson function to take a first guess step and then evaluate the integral using
    the adaptive Simpson's rule. Return the estimated value of the integrand and the number
    of steps taken to converge
    """
    [guess, h, fh] = simpson(f, a, b)
    [I, count] = adaptiveSimpson(f, a, b, h, guess, tol, nmax, 0)
    
    return I, count

#Function to integrate for electric field for thin, charged spherical shell
def f(u, z, R):
    return (z - R*u)/(R**2 + z**2 - 2*R*z*u)**(3/2)

#Integrate f to get electric field using my simpson integrator
def E_simpson(z, R, a, b, tol, nmax):
    #Store value of E at each pt
    E = []
    for u in z:
        #Catch div by zero errors
        f_int = lambda x : f(x, u, R)
        try:
            I, counts = simpsonIntegrator(f_int, a, b, tol, nmax)
        except:
            I = 0
        E.append(I)
    
    return E

#Integrate f to get electric field using the scipy quad integrator
def E_scipyQuad(z, R, a, b):
    E = []
    for u in z:
        f_int = lambda x : f(x, u, R)
        I, err = integrate.quad(f_int, a, b)
        E.append(I)
    
    return E

#Test the integrators by evaluating the electric field inside and outside of the shell
R=1
xp = np.linspace(0, 2*R, 1000)

E_myIntegrator = E_simpson(xp, R, -1, 1, 1e-7, 1000)
E_scipyIntegrator = E_scipyQuad(xp, R, -1, 1)

plt.plot(xp, E_myIntegrator, '-', label = "Simpson's Rule", color='red')
plt.plot(xp, E_scipyIntegrator, '--', label = "Scipy Quad", color='black')
plt.xlabel('Radius')
plt.ylabel('Electric Field Strength')
plt.title('Electric Field Due to Sphere of Radius {}'.format(R))
plt.legend()
    
    
    
    
    