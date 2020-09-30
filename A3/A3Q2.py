#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 23:02:23 2020

@author: noahlefrancois
"""

import numpy as np
import camb
from matplotlib import pyplot as plt

#Provided example function to get the power spectrum from CAMB
def get_spectrum(pars,lmax=2000):
    print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt
#Calculate the chi squ value of a fit
def chi_squared(data, fit, error):
    x = np.asarray(data)
    y = np.asarray(fit)
    error = np.asarray(error)  
    return sum((x-y)**2/error**2)


plt.ion()


#Load the power spectrum data and store the columns we need
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
multipole = wmap[:,0]; power = wmap[:,1]; errPower = wmap[:,2]

#Calculate the spectrum values for the specified fit params
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
cmb=get_spectrum(pars)
cmb = cmb[2:len(multipole)+2]

#Calculate & print the chi squ value of the fit
chi_sq = chi_squared(power, cmb, errPower)
print('Chi squared for the given fit parameters is ',chi_sq,' using the Gaussian, uncorrelated errors given')

#Plot the data and the fit
plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
plt.plot(multipole,power,'.',label='Data')

plt.plot(cmb,label='Fit')
plt.xlabel('Multipole Moment')
plt.ylabel('Power Spectrum')
plt.title('WMAP Satellite 9-year CMB Data')