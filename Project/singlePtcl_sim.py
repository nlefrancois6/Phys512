#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 20:08:59 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import NBody_systemInitialize as syst
import NBody_sim as nbod

#Switch to save the output plots
savePlots=True

#Number of ptcls
N = 10**0
#Grid dimensions (must be square for now)
LX = 64
LY = 64
size = (LY, LX)
#Time step size
h = 10**1
#Final time and number of time steps
T = 30
nsteps = int(T/h)

#Initialize particle position, velocity, and mass
x0 = np.array([[LY/2, LX/2]])
v0 = np.array([[0.,0.]])
m0 = [10 for t in range(N)]

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0)
#Initialize the solver for our system of N particles
sim = nbod.NBody_solver(size,system,h)

#Run the simulation

#Store the density and energy from each time-step
rho_store = np.zeros((nsteps+1, LY, LX))
E_store = np.zeros(nsteps)

rho_store[0,:,:] = sim.rho

#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:] = sim.rho
    E_store[t] = sim.E
    
#Plot the energy and save the frames for the video
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q1_energy.png')
    
#Initialize the figure
fig = plt.figure()
plt.pcolormesh(rho_store[0,:,:], cmap = cm.plasma)
plt.colorbar()

if savePlots:
    for t in range(nsteps):
        plt.title('Density Field at t = '+ str((t+1)*h))
        plt.pcolormesh(rho_store[t+1,:,:], cmap = cm.plasma)
        filename = 'Frames/Q1_frame'+str(t+1)+'.png'
        plt.savefig(filename)