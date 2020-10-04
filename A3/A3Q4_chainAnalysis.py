#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:48:29 2020

@author: noahlefrancois
"""

import numpy as np
import camb
from matplotlib import pyplot as plt

def get_spectrum(pars,lmax=2000, fixTau = False):
    if fixTau==False:
        #print('pars are ',pars)
        H0=pars[0]
        ombh2=pars[1]
        omch2=pars[2]
        tau=pars[3]
        As=pars[4]
        ns=pars[5]
    else:
        H0=pars[0]
        ombh2=pars[1]
        omch2=pars[2]
        tau=fixTau
        As=pars[3]
        ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

#Load data
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
multipole = wmap[:,0]; power = wmap[:,1]; errPower = wmap[:,2]
#Load MCMC outputs
chain = np.loadtxt('Q5_MCMC_outputChain.txt')
chi_sq = np.loadtxt('Q5_MCMC_outputChiSq.txt')
params = np.loadtxt('Q5_MCMC_outputParams.txt')
#parameter_labels = [r'H_0',r'$\omega_b h^2$',r'$\omega_c h^2',r'$\tau$',r'A_s','Slope']
parameter_labels = ['H_0','O_b h^2','O_c h^2','tau','A_s','Slope']

#Calculate the fit using the mcmc params
mcmcFit = get_spectrum(params, fixTau = False)[2:len(power)+2]

#Remove the zeros from rejected steps
chi_sq_comp = np.ma.masked_equal(chi_sq,0).compressed()
chain_comp = np.zeros((len(chi_sq_comp),len(params)))

#Plot the random walk of each parameter and mark the approximate end of the burn-in phase
burn_in_end = 225
f1 = plt.figure()
for i in range(len(params)):
    chain_comp[:,i] = np.ma.masked_equal(chain[:,i],0).compressed()
    ax = f1.add_subplot(6,1,i+1)
    ax.axhline(np.mean(chain_comp[:,i]),color='k', linestyle = ':')
    ax.axvline(burn_in_end, color='r')
    ax.plot(chain_comp[:,i])
    ax.set_ylabel(parameter_labels[i])
ax.set_xlabel('MCMC Steps')

#Plot the convergence of chi-squared and the fit
f2 = plt.figure()
ax = f2.add_subplot(2,1,1)
ax.axhline(np.mean(chi_sq_comp),color='k', linestyle = ':')
ax.axvline(burn_in_end, color='r')
ax.plot(chi_sq_comp)
ax.set_ylabel('Chi-Squared')
ax.set_xlabel('MCMC Steps')

ax = f2.add_subplot(2,1,2)
#ax.plot(multipole,power,'.',label='Data')
ax.errorbar(multipole,power,errPower,fmt='.',label='Data',zorder=1)
ax.plot(mcmcFit,label='MCMC Fit')
ax.set_xlabel('Multipole Moment')
ax.set_ylabel('Power Spectrum')
ax.set_title('WMAP Satellite 9-year CMB Data')
ax.legend()

    
    
    
