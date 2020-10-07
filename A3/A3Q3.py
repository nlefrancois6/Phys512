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
    dx = 0.01
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
        #print(dp)
        
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

parameter_labels = ['H_0','O_b h^2','O_c h^2','A_s','Slope']
storeDerivs = np.zeros((len(power),len(paramsFix)))
dx= 0.01

storeDerivs = partial_derivatives(power, get_spectrum, paramsFix, dx, tau)
for i in range(len(paramsFix)):
    plt.semilogy(abs(storeDerivs[:,i]),label='Parameter: {}'.format(parameter_labels[i]))

plt.xlabel('Multipole Moment')
plt.ylabel('Power Spectrum')
plt.title('Numerical Derivatives')
plt.legend(loc='upper right')
"""
This plot of the numerical derivative for each parameter across the data range shows that all of them
stay within a few orders of magnitude, indicating that we should be okay with this dx size and it 
should be able to reasonably capture the behaviour of each function.
"""

"""
I expect the errors to be lower when allowing tau to float since more of the parameter space is 
accessible to look for a minimum of chi-squared.

Output: 
    
Iteration  0 : 
Chi-Squared =  1588.237646582676 
Lambda =  1e-05 

Chi-Squared decreased. Lambda has been decreased.
Iteration  1 : 
 Chi-Squared =  1234.804081089522 
 Lambda =  1.0000000000000002e-06 

Chi-Squared decreased. Lambda has been decreased.

Iteration  2 : 
 Chi-Squared =  1227.936903372273 
 Lambda =  1.0000000000000002e-07 

Chi-Squared decreased. Lambda has been decreased.

Iteration  3 : 
 Chi-Squared =  1227.9356355996376 
 Lambda =  1.0000000000000004e-08 

Chi-Squared converged to  1227.9356339593712 
"""

#Fit with tau as a free parameter
paramsFree, covFree, errFree, chi_store_Free = Newt_Lev_Marq_minimizer(power, get_spectrum, pars_initialGuess_tauFree, errPower, max_iter, tau=False)
pFit_Free = get_spectrum(paramsFree)[2:len(power)+2]

"""
Findings: Chi-Squared is almost the same (1227.9 for fixed, 1227.8 for free) and errors are all
of the same magnitude. However, letting tau be a free parameter results in a much larger number
of iterations before the fit converges. This makes sense since there is an additional degree of 
freedom in this fit.

Output:

Chi-Squared decreased. Lambda has been decreased.

Iteration  0 : 
 Chi-Squared =  1588.237646582676 
 Lambda =  1e-05 

Chi-Squared decreased. Lambda has been decreased.

Iteration  1 : 
 Chi-Squared =  1235.5192579744582 
 Lambda =  1.0000000000000002e-06 

Chi-Squared decreased. Lambda has been decreased.

Iteration  2 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  1.0000000000000002e-07 

Chi-Squared increased. Lambda has been increased.

Iteration  3 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  1.0000000000000003e-05 

Chi-Squared increased. Lambda has been increased.

Iteration  4 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  0.0010000000000000002 

Chi-Squared increased. Lambda has been increased.

Iteration  5 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  0.10000000000000002 

Chi-Squared increased. Lambda has been increased.

Iteration  6 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  10.000000000000002 

Chi-Squared increased. Lambda has been increased.

Iteration  7 : 
 Chi-Squared =  1227.9836259688445 
 Lambda =  1000.0000000000002 

Chi-Squared decreased. Lambda has been decreased.

Iteration  8 : 
 Chi-Squared =  1227.9786348426355 
 Lambda =  100.00000000000003 

Chi-Squared decreased. Lambda has been decreased.

Iteration  9 : 
 Chi-Squared =  1227.972106442598 
 Lambda =  10.000000000000004 

Chi-Squared decreased. Lambda has been decreased.

Iteration  10 : 
 Chi-Squared =  1227.9302550685518 
 Lambda =  1.0000000000000004 

Chi-Squared decreased. Lambda has been decreased.

Iteration  11 : 
 Chi-Squared =  1227.876093263126 
 Lambda =  0.10000000000000005 

Chi-Squared decreased. Lambda has been decreased.

Iteration  12 : 
 Chi-Squared =  1227.8698416859336 
 Lambda =  0.010000000000000005 

Chi-Squared decreased. Lambda has been decreased.

Iteration  13 : 
 Chi-Squared =  1227.8682384894826 
 Lambda =  0.0010000000000000007 

Chi-Squared decreased. Lambda has been decreased.

Iteration  14 : 
 Chi-Squared =  1227.860095673268 
 Lambda =  0.00010000000000000007 


Iteration  15 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  1.0000000000000008e-05 

Chi-Squared increased. Lambda has been increased.

Iteration  16 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  0.0010000000000000007 

Chi-Squared increased. Lambda has been increased.

Iteration  17 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  0.10000000000000006 

Chi-Squared increased. Lambda has been increased.

Iteration  18 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  10.000000000000005 

Chi-Squared increased. Lambda has been increased.

Iteration  19 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  1000.0000000000006 

Chi-Squared increased. Lambda has been increased.

Iteration  20 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  100000.00000000006 

Chi-Squared increased. Lambda has been increased.

Iteration  21 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  10000000.000000006 

Chi-Squared increased. Lambda has been increased.

Iteration  22 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  1000000000.0000006 

Chi-Squared increased. Lambda has been increased.

Iteration  23 : 
 Chi-Squared =  1227.8196549427323 
 Lambda =  100000000000.00006 

Chi-Squared converged to  1227.8196548828296 

"""

#Save the results of both fits to a txt file
np.savetxt('Q3_fixedTau_params.txt', paramsFix)
np.savetxt('Q3_fixedTau_err.txt', errFix)
np.savetxt('Q3_fixedTau_cov.txt', covFix)

np.savetxt('Q3_freeTau_params.txt', paramsFree)
np.savetxt('Q3_freeTau_err.txt', errFree)
np.savetxt('Q3_freeTau_cov.txt', covFree)


#Plot the data and the fit, as well as a convergence plot for Chi-Squared

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
