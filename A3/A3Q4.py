#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:20:29 2020

@author: noahlefrancois
"""

import numpy as np
import camb
from matplotlib import pyplot as plt

#Provided example function to get the power spectrum from CAMB
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

def run_MCMC(y_data, func, params, err_y, chain_length, p_cov, tau):
     j = 0
     chains_list = np.zeros([chain_length,len(params)])
     chi_sq_list = np.zeros(chain_length)
     y_guess = func(params, fixTau=tau)[2:len(y_data)+2]
     chi_sq = chi_squared(y_data, y_guess, err_y)
     lamb = 0.5
     for i in range(chain_length):
         print('Step ', i, ': Chi-Squared=', chi_sq)
         r = np.linalg.cholesky(p_cov)
         d = np.dot(r, np.random.randn(r.shape[0]))
         params_guess = params + d*lamb
         #Only accept guesses with positive tau values
         if params_guess[3]>0:
             y_guess_update = func(params_guess, fixTau=tau)[2:len(y_data)+2]
             chi_sq_update = chi_squared(y_data, y_guess_update, err_y)
             delta_Chi = chi_sq_update - chi_sq
             prob_accept = np.exp(-0.5*delta_Chi)
             accept_step = np.random.rand(1) < prob_accept
             if accept_step == True:
                 print('Step Accepted. Params Updated.')
                 j += 1
                 params = params_guess
                 y_guess = y_guess_update
                 chi_sq = chi_sq_update
             else:
                 print('Step Rejected')
             chains_list[i,:] = params
             chi_sq_list[i] = chi_sq
         else:
             print('Negative tau value rejected')
     return chains_list, chi_sq_list, params
     

#Load the power spectrum data and store the columns we need
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
multipole = wmap[:,0]; power = wmap[:,1]; errPower = wmap[:,2]

#Specify our initial guess and other settings
tau = False
pars_initialGuess=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
#Run-time is approx 10 steps per minute on Noah's laptop
chainLength = 100
#Need to replace this with reading the matrix out of my saved txt file from Q3
p_cov=np.loadtxt('Q3_fixedTau_cov.txt')
"""
p_cov = np.asarray([[8.27963540e+00,1.38059605e-03,-1.50966395e-02,2.31683686e-01,8.17793756e-10,4.25805390e-02],[1.38059605e-03,4.47664949e-07,-1.64546691e-06,5.76110907e-05,2.19546969e-13,1.10298580e-05]
 ,[-1.50966395e-02,-1.64546691e-06,3.43333369e-05,-3.77088120e-04,-1.24887877e-12,-6.36488736e-05],[2.31683686e-01,5.76110907e-05,-3.77088120e-04,2.07377984e-02,7.84029301e-11,1.80406820e-03]
 ,[8.17793756e-10,2.19546969e-13,-1.24887877e-12,7.84029301e-11,2.97761352e-19,6.77396391e-12],[4.25805390e-02,1.10298580e-05,-6.36488736e-05,1.80406820e-03,6.77396391e-12,3.40209983e-04]])
"""

#Run the MCMC
[chain, chi_sq, params]=run_MCMC(power, get_spectrum, pars_initialGuess, errPower, chainLength, p_cov, tau)

np.savetxt('MCMC_outputChain', chain)
np.savetxt('MCMC_outputChiSq', chi_sq)
np.savetxt('MCMC_outputParams', params)




