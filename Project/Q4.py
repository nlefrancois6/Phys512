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

Scipy fft in NBody_sim:
    Size 128, t = 0.0034177303314208984
    Size 512, t = 0.022250890731811523
    useful, will make this change
    
Scipy fft in NBody_systemInitialize (timed initialization):
    Size 128, numpy: 0.5634071826934814
    Size 128, scipy: 0.5464169979095459
    Size 512, numpy: 0.6636438369750977
    Size 512, scipy: 0.5789480209350586
    inconclusive, will not make this change
    Also only applies with cmass=True
"""

#Number of ptcls
N = 10**4
#Grid dimensions (must be square for now)
LX = 512
LY = 512
size = (LY, LX)
#Time step size
h = 10**1
#Final time and number of time steps
T = 200
nsteps = int(T/h)

#Initialize particle position, velocity, and mass
x0 = None
v0 = None
#v0 = np.array([[0,0] for t in range(N)]) #Initially all at rest IC
m = 1/N
m0 = [m for t in range(N)]
cmass = False

#Initialize the system of N particles
t1 = time.time()
system = syst.Nparticle_system(N, size, m0, set_x0 = x0, set_v0 = v0, soft=0.1, boundaryCondition='Periodic', cosmology_mass=cmass)
t2 = time.time()
print(t2-t1)

#Initialize the simulation for our system of N particles
sim = nbod.NBody_solver(size,system,h, cosmology_mass=cmass)

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
plt.pcolormesh(sim.rho, cmap = cm.plasma)
plt.colorbar()

#Run the simulation and generate the animation from it
anim = animation.FuncAnimation(fig, animate, frames = nsteps, interval = h, blit = False)
plt.show()
#anim.save('Part3_periodic.gif', writer='imagemagick')

#anim.save('Test.gif', writer='imagemagick')
