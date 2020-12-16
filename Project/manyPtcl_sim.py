#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:52:45 2020
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
N = 10**5
#Grid dimensions (must be square for now)
LX = 512
LY = 512
size = (LY, LX)
#Time step size
h = 10**2
#Final time and number of time steps
T = 10000
nsteps = int(T/h)

soften = 0.8
BC = 'Periodic'

#Initialize particle position, velocity, and mass
x0 = None
v0 = None
m = 1/N
m0 = [m for t in range(N)]
cmass = False

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, boundaryCondition = BC, soft=soften, cosmology_mass=cmass)


#Initialize the simulation for our system of N particles
#may just need to not use boundaryCondition here if I can't figure out how to fix it :/
sim = nbod.NBody_solver(size,system,h, soft = soften, cosmology_mass=cmass, boundaryCondition = BC)

#Run the simulation

#Store the density and energy from each time-step
rho_store = np.zeros((nsteps+1, LY, LX))
    
E_store = np.zeros(nsteps)

if BC == 'Periodic':
    rho_store[0,:,:] = sim.rho
if BC == 'Non-Periodic':
    rho_store[0,:,:] = sim.rho[:LY,:LX]
    
#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    if BC == 'Periodic':
        rho_store[t+1,:,:] = sim.rho
    if BC == 'Non-Periodic':
        rho_store[t+1,:,:] = sim.rho[:LY,:LX]
    E_store[t] = sim.E


#Plot the energy and save the density frames
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q3_energy_P_soft0p8_longT.png')
    
#Define the function used to update the animation at each frame
def animate(i):
    #Only update the plot for integer t
    if i*h == i*h//1:
        #global rho_store
        plt.title('Density Field at t = '+ str(i*h))
        #This is where new data is inserted into the plot.
        plt.pcolormesh(rho_store[i,:,:], cmap = cm.plasma)
        #plt.pause(0.1)
#Initialize the figure
fig = plt.figure()
plt.pcolormesh(rho_store[0,:,:], cmap = cm.plasma)
plt.colorbar()

if savePlots:
    for t in range(nsteps):
        print('Saved t = ',t+1)
        plt.title('Density Field at t = '+ str((t+1)*h))
        plt.pcolormesh(rho_store[t+1,:,:], cmap = cm.plasma)
        filename = 'Frames/Q3_NP_frame'+str(t+1)+'.png'
        plt.savefig(filename)