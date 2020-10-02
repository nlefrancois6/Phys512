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

"""
Make a small change in each parameter and get the function value at that new location in
order to compute the local gradient. Use central differencing scheme dy/dp = (y_{n+1}-y_{n-1})/(2dp)
"""  
def partial_derivatives(x, func, p, dx, tau):
    grad = np.zeros([len(x), len(p)])

    for i in range(len(p)):
        p_dx = p.copy()
        dp = p_dx[i]*dx
        
        p_dx[i] = p_dx[i] + dp
        y_forward = func(p_dx, fixTau = tau)[2:len(x)+2]
        p_dx[i] = p_dx[i] - 2*dp
        y_backward = func(p_dx, fixTau = tau)[2:len(x)+2]
        
        dydp_i = (y_forward - y_backward)/(2*dp)
        grad[:,i] = np.asarray(dydp_i)
    return grad        
    
def Newt_Lev_Marq_minimizer(y_data, func, p_guess, err_y, max_iter, tau=False):
    """
    Find the optimal parameters (p) to fit y_data to the model func with fixed tau
    """
    chi_sq = 1e5
    lamb = 1e-4
    dx = 0.001
    grad = partial_derivatives(y_data, func, p_guess, dx, tau)
    chi_store = np.zeros(1)
    p_update = p_guess.copy()
    for i in range(max_iter):
        y_guess = func(p_update, fixTau = tau)[2:len(y_data)+2]
        chi_sq_update = chi_squared(y_data, y_guess, err_y)
        chi_store = np.append(chi_store, [chi_sq_update])
        delta_Chi = chi_sq - chi_sq_update
        if delta_Chi<dx and delta_Chi>0:
            print('Chi-Squared converged to ', chi_sq_update, '\n')
            p_guess = p_update.copy()
            grad = partial_derivatives(y_data, func, p_guess, dx, tau)
            N = np.diag(1/err_y**2)
            cov = np.dot(grad.transpose(),N)
            cov = np.dot(cov,grad)
            cov = np.linalg.inv(cov)
            err = np.sqrt(np.diag(cov))     
            break
        elif chi_sq_update > chi_sq:
            print('Chi-Squared increased. Lambda has been increased.')
            lamb = lamb*1e2
            p_update = p_guess.copy()
        else:
            print('Chi-Squared decreased. Lambda has been decreased.')
            chi_sq = chi_sq_update
            p_guess = p_update.copy()
            lamb = lamb*1e-1
            grad = partial_derivatives(y_data, func, p_guess, dx, tau)
        res = np.matrix(y_data - y_guess).transpose()
        grad = np.matrix(grad)
        a = grad.transpose()*np.diag(1/err_y**2)*grad
        A = a + lamb*np.diag(np.diag(a))
        B = grad.transpose()*np.diag(1/err_y**2)*res
        dp= np.linalg.inv(A)*B
        print(dp)
        
        for j in range(len(p_guess)):
            p_update[j] = p_update[j] + dp[j]
        print('Iteration ', i, ': \n', 'Chi-Squared = ', chi_sq, '\n Lambda = ', lamb, '\n')
    #N = np.diag(1/err_y**2)
    #cov = np.linalg.pinv(np.dot(np.dot(grad.transpose(),N),grad))
    #err = np.sqrt(np.diag(cov))
    #print('Final fit has Chi-Squared value of ', chi_sq)
    return p_guess, cov, err, chi_store
        
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

#Calculate the spectrum values using the specified fit params as an initial guess
tau = 0.05
pars_initialGuess_tauFree=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars_initialGuess_tauFixed=np.asarray([65,0.02,0.1,2e-9,0.96])
max_iter = 100

#Fit with tau=0.05 fixed
paramsFix, covFix, errFix, chi_store_Fix = Newt_Lev_Marq_minimizer(power, get_spectrum, pars_initialGuess_tauFixed, errPower, max_iter, tau=0.05)
pFit_Fix = get_spectrum(paramsFix, fixTau = 0.05)[2:len(power)+2]

#Fit with tau as a free parameter
paramsFree, covFree, errFree, chi_store_Free = Newt_Lev_Marq_minimizer(power, get_spectrum, pars_initialGuess_tauFree, errPower, max_iter, tau=False)
pFit_Free = get_spectrum(paramsFree)[2:len(power)+2]

#Save the results of both fits to a txt file
np.savetxt('Q3_fixedTau_params.txt', paramsFix)
np.savetxt('Q3_fixedTau_err.txt', errFix)
np.savetxt('Q3_fixedTau_cov.txt', covFix)

np.savetxt('Q3_freeTau_params.txt', paramsFree)
np.savetxt('Q3_freeTau_err.txt', errFree)
np.savetxt('Q3_freeTau_cov.txt', covFree)
"""
saveFile = open('Q3_fixedTau_fit.txt', 'w')
np.savetxt(saveFile, paramsFix)
np.savetxt(saveFile, errFix)
np.savetxt(saveFile, covFix)
saveFile.close()

saveFile = open('Q3_freeTau_fit.txt', 'w')
np.savetxt(saveFile, paramsFree)
np.savetxt(saveFile, errFree)
np.savetxt(saveFile, covFree)
saveFile.close()
"""

#Calculate & print the chi squ value of the fit
#chi_sq = chi_squared(power, cmb, errPower)
#print('Chi squared for the given fit parameters is ',chi_sq,' using the Gaussian, uncorrelated errors given')

#Plot the data and the fit
plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
plt.plot(multipole,power,'.',label='Data')

plt.plot(pFit_Fix,label='Fit, Fixed Tau')
plt.plot(pFit_Free,label='Fit, Free Tau')
plt.xlabel('Multipole Moment')
plt.ylabel('Power Spectrum')
plt.title('WMAP Satellite 9-year CMB Data')
plt.legend()

plt.figure()
plt.plot(chi_store_Fix[1:], label='Fixed Tau')
plt.plot(chi_store_Free[1:], label='Free Tau')
plt.xlabel('Iteration')
plt.ylabel('Chi-Squared')
plt.title('Chi-Squared Convergence')
plt.legend()
