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
from mpl_toolkits import mplot3d

import NBody_systemInitialize3D as syst
import NBody_sim3D as nbod

#Switch to save the output plots
savePlots=True

#Number of ptcls
N = 10**4
#Grid dimensions (must be square for now)
LX = 64  
LY = 64
LZ = 64
size = (LY, LX, LZ)
#Time step size
h = 10**2
#Final time and number of time steps
T = 5000
nsteps = int(T/h)

soften = 0.8
BC = 'Periodic'

#Initialize particle position, velocity, and mass
x0 = None
v0 = None
m = 1/N
m0 = [m for t in range(N)]
cmass = True

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, v0_max = 0.01, boundaryCondition = BC, soft=soften, cosmology_mass=cmass)


#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, soft = soften, cosmology_mass=cmass, boundaryCondition = BC)

#Run the simulation

#We'll want to store the density field, energy, and frame from each time-step
if BC == 'Non-Periodic':
    LY2 = 2*LY; LX2 = 2*LX; LZ2 = 2*LZ
    rho_store = np.zeros((nsteps+1, LY2, LX2, LZ2))
    #rho_store = np.zeros((nsteps+1, LY, LX, LZ))
else:
    rho_store = np.zeros((nsteps+1, LY, LX, LZ))
    
E_store = np.zeros(nsteps)

rho_store[0,:,:,:] = sim.rho

xs_store = np.zeros((N,nsteps+1)); ys_store = np.zeros((N,nsteps+1)); zs_store = np.zeros((N,nsteps+1))
xs_store[:,0] = sim.x[:,0]; ys_store[:,0] = sim.x[:,1]; zs_store[:,0] = sim.x[:,2]


#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:,:] = sim.rho
    E_store[t] = sim.E
    xs_store[:,t+1] = sim.x[:,0]; ys_store[:,t+1] = sim.x[:,1]; zs_store[:,t+1] = sim.x[:,2]
 
"""
#Save and plot the energy and the video frames
"""
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q4_3D_energy.png')

if savePlots:
    for t in range(nsteps):
        ax = plt.axes(projection ="3d")
        plt.title("Density Field at t = "+ str(t*h))
        ax.scatter3D(xs_store[:,t], ys_store[:,t], zs_store[:,t], color = "green", marker='o', s=1)
        filename = 'Frames/Q_3D_frame'+str(t+1)+'.png'
        plt.savefig(filename)


