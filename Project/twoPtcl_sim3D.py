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
from mpl_toolkits import mplot3d

import NBody_systemInitialize3D as syst
import NBody_sim3D as nbod

#Switch to save the output plots
savePlots=True

#Number of ptcls
N = 2
#Grid dimensions (must be square for now)
LX = 128
LY = 128
LZ = 128
size = (LY, LX, LZ)
#Time step size
h = 10**1
#Final time and number of time steps
T = 500
nsteps = int(T/h)

soften = 0.8
BC = 'Periodic'

#Initialize particle position, velocity, and mass
x0 = np.array([[LY/2, LX/2 - 5, LZ/2],[LY/2, LX/2 + 5, LZ/2]]) #Note first value is x_y, second is x_x
v0 = np.array([[0.5,0, 0],[-0.5,0,0]]) #Ptcl initially at rest. Note first value is v_y, second is v_y
m0 = [100 for t in range(N)]
#cmass = False

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, soft=soften, boundaryCondition = BC)
#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, soft=soften, boundaryCondition = BC)

#Run the simulation

#We'll want to store the density field, energy, and frame from each time-step
if BC == 'Non-Periodic':
    LY2 = 2*LY; LX2 = 2*LX; LZ2 = 2*LZ
    rho_store = np.zeros((nsteps+1, LY2, LX2, LZ2))
else:
    rho_store = np.zeros((nsteps+1, LY, LX, LZ))
    
E_store = np.zeros(nsteps)

rho_store[0,:,:,:] = sim.rho

xs_store = np.zeros((N,nsteps+1)); ys_store = np.zeros((N,nsteps+1)); zs_store = np.zeros((N,nsteps+1))
xs_store[:,0] = sim.x[:,0]; ys_store[:,0] = sim.x[:,1]; zs_store[:,0] = sim.x[:,2]

"""
Method 2 (seems to be the better option):
Makes an animation AFTER the simulation runs & replays it infinitely
Should theoretically be easy to save anim after but it's giving a 'list index out of range' error
"""
#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:,:] = sim.rho
    E_store[t] = sim.E
    xs_store[:,t+1] = sim.x[:,0]; ys_store[:,t+1] = sim.x[:,1]; zs_store[:,t+1] = sim.x[:,2]
    
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    if savePlots:
        plt.savefig('Energy/Q2_3D_energy.png')
    
#Define the function used to update the animation at each frame
def animate(i):
    #Only update the plot for integer t
    plt.title('Density Field at t = '+ str(i*h))
    #This is where new data is inserted into the plot.
    plt.pcolormesh(rho_store[i,:,:,0], cmap = cm.plasma)
    #plt.pause(0.01)
"""
#Initialize the figure
fig = plt.figure()
plt.pcolormesh(rho_store[0,:,:,0], cmap = cm.plasma)
plt.colorbar()
"""

if savePlots:
    for t in range(nsteps):
        ax = plt.axes(projection ="3d")
        plt.title("Density Field at t = "+ str(t*h))
        ax.scatter3D(xs_store[:,t], ys_store[:,t], zs_store[:,t], color = "green", marker='o', s=10)
        filename = 'Frames/Q2_3D_frame'+str(t+1)+'.png'
        plt.savefig(filename)

"""
#Run the simulation and generate the animation from it
anim = animation.FuncAnimation(fig, animate, frames = nsteps, interval = h, blit = False)
plt.show()
#anim.save('Part2.gif', writer='imagemagick')
"""