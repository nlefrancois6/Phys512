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


#Number of ptcls
N = 2
#Grid dimensions (must be square for now)
LX = 512
LY = 512
size = (LY, LX)
#Time step size
h = 10**1
#Final time and number of time steps
T = 300
nsteps = int(T/h)

#Initialize particle position, velocity, and mass
x0 = np.array([[LY/2, LX/2 - 10],[LY/2, LX/2 + 10]]) #Note first value is x_y, second is x_x
v0 = np.array([[0.1,0],[-0.1,0]]) #Ptcl initially at rest. Note first value is v_y, second is v_y
m0 = [10 for t in range(N)]

#Initialize the system of N particles
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, soft=0.1, boundaryCondition = 'Non-Periodic')
#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, soft=0.1)

#Run the simulation

#We'll want to store the density field, energy, and frame from each time-step
rho_store = np.zeros((nsteps+1, LY, LX))
E_store = np.zeros(nsteps)

rho_store[0,:,:] = sim.rho

"""
Method 2 (seems to be the better option):
Makes an animation AFTER the simulation runs & replays it infinitely
Should theoretically be easy to save anim after but it's giving a 'list index out of range' error
"""
#Use advance_timeStep() nsteps-many times
for t in range(nsteps):
    #Store density field and energy
    E, x = sim.advance_timeStep()
    rho_store[t+1,:,:] = sim.rho
    E_store[t] = sim.E
    
showEnergyPlot = True
if showEnergyPlot == True:
    plt.figure()
    plt.plot(E_store)
    plt.title('System Total Energy')
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    
#Define the function used to update the animation at each frame
def animate(i):
    #Only update the plot for integer t
    plt.title('Density Field at t = '+ str(i*h))
    #This is where new data is inserted into the plot.
    plt.pcolormesh(rho_store[i,:,:], cmap = cm.plasma)
    #plt.pause(0.01)
#Initialize the figure
fig = plt.figure()
plt.pcolormesh(sim.rho, cmap = cm.plasma)
plt.colorbar()

#Run the simulation and generate the animation from it
anim = animation.FuncAnimation(fig, animate, frames = nsteps, interval = h, blit = False)
plt.show()
#anim.save('Part2.gif', writer='imagemagick')

