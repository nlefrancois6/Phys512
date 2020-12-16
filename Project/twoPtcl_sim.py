#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:06:45 2020
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
N = 2
#Grid dimensions (must be square for now)
LX = 128
LY = 128
size = (LY, LX)
#Time step size
h = 10**1
#Final time and number of time steps
T = 50
nsteps = int(T/h)

soften = 1.5
BC = 'Periodic'

#Initialize particle position, velocity, and mass
x0 = np.array([[LY/2, LX/2 - 10],[LY/2, LX/2 + 10]]) #Note first value is x_y, second is x_x
v0 = np.array([[0.1,0],[-0.1,0]]) #Ptcl initially at rest. Note first value is v_y, second is v_y
m0 = [10 for t in range(N)]

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, soft=soften, boundaryCondition = BC)
#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, soft=soften, boundaryCondition = BC)

#Run the simulation

#We'll want to store the density field, energy, and frame from each time-step
if BC == 'Non-Periodic':
    LY2 = 2*LY; LX2 = 2*LX
    rho_store = np.zeros((nsteps+1, LY2, LX2))
else:
    rho_store = np.zeros((nsteps+1, LY, LX))
    
E_store = np.zeros(nsteps)

rho_store[0,:,:] = sim.rho


#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:] = sim.rho
    E_store[t] = sim.E

#Plot the energy and save the video frames
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q2_energy.png')
    
#Initialize the figure
fig = plt.figure()
plt.pcolormesh(rho_store[0,:,:], cmap = cm.plasma)
plt.colorbar()

if savePlots:
    for t in range(nsteps):
        plt.title('Density Field at t = '+ str((t+1)*h))
        plt.pcolormesh(rho_store[t+1,:,:], cmap = cm.plasma)
        filename = 'Frames/Q2_frame'+str(t+1)+'.png'
        plt.savefig(filename)
