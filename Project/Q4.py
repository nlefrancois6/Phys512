#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:30:26 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import NBody_systemInitialize as syst
import NBody_sim as nbod

import time

#Switch to save the output plots
savePlots=True

"""
Performance (1 Time Step):
N = 10^4
Baseline: 
    Size 128, t = 0.00964212417602539
    Size 512, t = 0.05220389366149902
    
fast_histogram:
    Size 128, t = 0.0043141841888427734
    Size 512, t = 0.04168510437011719
    useful, will make this change
    However, does not support 3D histogram so only use it for 2D

Scipy fft in NBody_sim:
    Size 128, t = 0.0034177303314208984
    Size 512, t = 0.022250890731811523
    useful, will make this change
    
Scipy fft in NBody_systemInitialize (timed initialization):
    Size 128, numpy: 0.5634071826934814
    Size 128, scipy: 0.5464169979095459
    Size 512, numpy: 0.6636438369750977
    Size 512, scipy: 0.5789480209350586
    Inconclusive, so I will not make this change
    Also only applies with cmass=True anyways
"""

#Number of ptcls
N = 10**4
#Grid dimensions (must be square for now)
LX = 128
LY = 128
size = (LY, LX)
#Time step size
h = 10**2
#Final time and number of time steps
T = 2000
nsteps = int(T/h)

soften = 10
BC = 'Periodic'

#Initialize particle position, velocity, and mass
x0 = None
v0 = None
m = 50
m0 = [m for t in range(N)]
cmass = True

#Initialize the system of N particles
t1 = time.time()
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, soft=soften, boundaryCondition=BC, cosmology_mass=cmass)
t2 = time.time()
print(t2-t1)

#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, boundaryCondition=BC, soft = soften, cosmology_mass=cmass)

#Store the density and energy from each time-step
rho_store = np.zeros((nsteps+1, LY, LX))
E_store = np.zeros(nsteps)

rho_store[0,:,:] = sim.rho

#Use advance_timeStep() nsteps-many times
t3 = time.time()
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:] = sim.rho
    E_store[t] = sim.E    
t4 = time.time()
print(t4-t3)


showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q4_energy.png')

#Initialize the figure
fig = plt.figure()
plt.pcolormesh(rho_store[0,:,:], cmap = cm.plasma)
plt.colorbar()

if savePlots:
    for t in range(nsteps):
        print('Saved t = ',t+1)
        plt.title('Density Field at t = '+ str((t+1)*h))
        plt.pcolormesh(rho_store[t+1,:,:], cmap = cm.plasma)
        filename = 'Frames/Q4_frame'+str(t+1)+'.png'
        plt.savefig(filename)