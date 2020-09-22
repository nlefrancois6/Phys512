#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:37:19 2020

@author: noahlefrancois
"""

import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

def fun(x,y,half_life=[1,1e-5]):
    #let's do a 2-state radioactive decay
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=y[0]/half_life[0]-y[1]/half_life[1]
    dydx[2]=y[1]/half_life[1]
    return dydx
#pre-store conversion rates to seconds for convenience
year = 3.154e7
billion_year = year*1e9
day = 86400
hour = 3600
minute = 60
usecond = 1e-6

#Store the half-lives of each isotope in the chain
U238_hl = [4.468*billion_year, 24.1*day, 6.7*hour, 245500*year, 75380*year, 1600*year, 3.8235*day, 3.1*minute, 26.8*minute, 19.9*minute, 164.3*usecond, 22.3*year, 5.015*year, 138.376*day]

#Define the system of ODEs for a list of half-lives
def decay_chain(x, y, half_life=U238_hl):
    dydx = np.zeros(len(half_life)+1)
    dydx[0] = -y[0]/half_life[0]
    for i in range(1, len(half_life)):
        dydx[i] = y[i-1]/half_life[i-1] - y[i]/half_life[i]
    dydx[-1] = y[-2]/half_life[-2]
    
    return dydx

#Set initial conditions and boundary conditions
y0=np.asarray([1] + [0]*len(U238_hl)) 
x0=0
x1=1e1*billion_year

#Run and time the rk4 solver and the Radau solver we saw in class
#Takes 580 seconds for x1=1e3 vs 0.0088 seconds with Radau solver
"""
t1=time.time();
ans_rk4=integrate.solve_ivp(decay_chain,[x0,x1],y0);
t2=time.time();
print('took ',ans_rk4.nfev,' evaluations and ',t2-t1,' seconds to solve with RK4.')
"""

t1=time.time()
ans_Radau=integrate.solve_ivp(decay_chain,[x0,x1],y0,method='Radau')
t2=time.time()
print('Radau took ',ans_Radau.nfev,' evaluations and ',t2-t1,' seconds')
print('final values were ',ans_Radau.y[0,-1])

"""
#Tested BDF and LSODA solvers, BDF was slightly slower (0.035 vs 0.030 sec) for x1 = billion_year, LSODA failed to converge for this time
#Their output was roughly equal for this time, U238 = 0.803 vs 0.799
#----> So I chose Radau as the best solver for this problem

t1=time.time()
ans_BDF=integrate.solve_ivp(decay_chain,[x0,x1],y0,method='BDF')
t2=time.time()
print('BDF took ',ans_BDF.nfev,' evaluations and ',t2-t1,' seconds')
print('final values were ',ans_BDF.y[0,-1])

t1=time.time()
ans_LSODA=integrate.solve_ivp(decay_chain,[x0,x1],y0,method='LSODA')
t2=time.time()
print('LSODA took ',ans_LSODA.nfev,' evaluations and ',t2-t1,' seconds')
print('final values were ',ans_LSODA.y[0,-1])
"""


#Ratio of Pb-206 to U-238 and Th-230 to U-234
ratio_Pb_U = ans_Radau.y[14,:]/ans_Radau.y[0,:]
ratio_Th_U = ans_Radau.y[4,:]/ans_Radau.y[3,:]

plt.figure()
plt.plot(ans_Radau.t[0:-1], ratio_Pb_U[0:-1])
plt.title('Ratio of Pb-206 to U-238')
plt.xlabel('Time (s)')
plt.ylabel('Ratio')

plt.figure()
plt.plot(ans_Radau.t[0:-1], ratio_Th_U[0:-1])
plt.title('Ratio of Th-230 to U-234')
plt.xlabel('Time (s)')
plt.ylabel('Ratio')



"""
plt.figure()
#plt.plot(ans_Radau.t, ans_Radau.y[0,:], label = 'U-238')
#plt.plot(ans_Radau.t, ans_Radau.y[1,:], label = 'Th-234')
#plt.plot(ans_Radau.t, ans_Radau.y[2,:], label = 'Pr-234')
#plt.plot(ans_Radau.t, ans_Radau.y[3,:], label = 'U-234')
#plt.plot(ans_Radau.t, ans_Radau.y[4,:], label = 'Th-230')
#plt.plot(ans_Radau.t, ans_Radau.y[5,:], label = 'Rm-226')
#plt.plot(ans_Radau.t, ans_Radau.y[6,:], label = 'Rn-222')
#plt.plot(ans_Radau.t, ans_Radau.y[7,:], label = 'Po-218')
#plt.plot(ans_Radau.t, ans_Radau.y[8,:], label = 'Pb-214')
#plt.plot(ans_Radau.t, ans_Radau.y[9,:], label = 'Bi-214')
#plt.plot(ans_Radau.t, ans_Radau.y[10,:], label = 'Po-214')
#plt.plot(ans_Radau.t, ans_Radau.y[11,:], label = 'Pb-210')
#plt.plot(ans_Radau.t, ans_Radau.y[12,:], label = 'Bi-210')
#plt.plot(ans_Radau.t, ans_Radau.y[13,:], label = 'Po-210')
plt.plot(ans_Radau.t, ans_Radau.y[14,:], label = 'Pb-206')
plt.legend()
plt.yscale('log')
"""
