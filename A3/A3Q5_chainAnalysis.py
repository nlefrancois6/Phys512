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
burn_in_end = 250
params_mean_burnedIn = np.zeros((len(params),1))
params_std_burnedIn = np.zeros((len(params),1))
f1 = plt.figure()
for i in range(len(params)):
    chain_comp[:,i] = np.ma.masked_equal(chain[:,i],0).compressed()
    params_mean_burnedIn[i,0] = np.mean(chain_comp[burn_in_end:,i])
    params_std_burnedIn[i,0] = np.std(chain_comp[burn_in_end:,i])
    ax = f1.add_subplot(6,1,i+1)
    ax.axhline(params_mean_burnedIn[i,0],color='k', linestyle = ':')
    ax.axvline(burn_in_end, color='r')
    ax.plot(chain_comp[:,i])
    ax.set_ylabel(parameter_labels[i])
    print(parameter_labels[i],' fit value ', f'{params_mean_burnedIn[i,0]:.3}','+/-', f'{params_std_burnedIn[i,0]:.3}')
ax.set_xlabel('MCMC Steps')

chi_sq_mean = np.mean(chi_sq_comp[burn_in_end:])
chi_sq_std = np.std(chi_sq_comp[burn_in_end:])
print('Chi-Squared fit value ', f'{chi_sq_mean:.3}','+/-', f'{chi_sq_std:.3}')

#Plot the convergence of chi-squared and the fit
f2 = plt.figure()
ax = f2.add_subplot(2,1,1)
ax.axhline(chi_sq_mean,color='k', linestyle = ':')
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

"""
Output:
H_0  fit value  69.1 +/- 1.94
O_b h^2  fit value  0.0224 +/- 0.000446
O_c h^2  fit value  0.114 +/- 0.0047
tau  fit value  0.0537 +/- 0.0132
A_s  fit value  2.06e-09 +/- 6.48e-11
Slope  fit value  0.968 +/- 0.0117
Chi-Squared fit value  1.23e+03 +/- 2.67

I think that the chain has converged because after the burn-in phase each of the parameters 
fluctuates roughly uniformly about the post-burn-in mean value as shown by the black dotted line.
Similarly, the chi-squared value fluctuates very close to the post-burn-in mean value as shown by
the relatively small standard deviation of 1230 +/- 2.67.

The fit results are mostly similar to those of the full chain in Q4, with tau being the only significantly
different parameter between the two chains. However, the error bars are smaller by factors of ~2 for each
of the parameters. 
The number of samples is also smaller for this question as more steps were rejected out of the
total runtime of 5000 steps due to the filtering of tau to be withing the measurement prior.
This means that giving the prior information for tau significantly reduces both the total number of steps
and the number of accepted steps necessary for the chain to converge.
"""
    
    
