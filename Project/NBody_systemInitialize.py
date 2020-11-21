#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:01:33 2020

@author: noahlefrancois
"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt

class ptcl:
    def __init__(self, m, x, y, vx=0, vy=0):
        """
        Initialize a particle which has mass, position, momentum
        """
        self.m = m
        self.x = (x,y)
        self.v = (vx, vy)

class Nparticle_system:
    def __init__(self, N, size, set_m0, set_x0 = None, set_v0 = None, boundaryCondition = 'Periodic', soft = None, cosmos = False):
        """
        Initialize a system of N particles with grid dimensions specified by 'size' 
        Can specify initial values for mass, position and velocity
        """
        self.N = N
        self.size = size
        self.cosmos = cosmos
        self.BC = boundaryCondition
        
        #Initialize x, v, m and assign these ICs to a list of particles
        self.get_x0(set_x0)
        self.get_v0(1, set_v0)
        self.get_m0(set_m0, soft)
        self.particles = np.asarray([ptcl(m,x[0],x[1],vx=v[0],vy=v[1]) for m,x,v in zip(self.m,self.x,self.v)])
        
        #Not sure if these lines are redundant or not since they should be set already in the block above
        """
        self.x = np.asarray([self.particles[i].x for i in range(N)])
        self.v = np.asarray([self.particles[i].v for i in range(N)])
        self.get_m0(set_m0, soft)
        """
    
    
    def get_x0(self, set_x0 = None):
        """
        Get the initial values for position either set directly by the user or by random generation
        """
        if self.BC == 'Periodic':
            IC = []
            if set_x0 is None:
                #If no initial positions set, we don't need to store any (all will be generated below)
                numICs_set = 0
            else:
                #If the user has set some (or all) of the initial positions, get those and store them
                numICs_set = len(set_x0)
                for i in range(numICs_set):
                    IC.append((set_x0[i][0], set_x0[i][1]))
            #Generate and store the remaining un-set initial positions
            for j in range(self.N - numICs_set):
                IC.append((rn.random()*(self.size[0]-1), rn.random()*(self.size[1]-1)))
            
            self.x = np.asarray(IC)
    
    def get_v0(self, speedLimit, set_v0 = None):
        """
        Get the initial values for velocity either set directly by the user or by random generation
        """     
        IC = []
        if set_v0 is None:
            #If no initial velocities set, we don't need to store any (all will be generated below)
            numICs_set = 0
        else:
            #If the user has set some (or all) of the initial positions, get those and store them
            numICs_set = len(set_v0)
            for i in range(numICs_set):
                IC.append((set_v0[i][0], set_v0[i][1]))
        #Generate and store the remaining un-set initial positions
        #Currently using uniform dist, could change to gaussian or something else
        for j in range(self.N - numICs_set):
            IC.append((rn.random()*2*speedLimit-speedLimit, rn.random()*2*speedLimit-speedLimit))
        
        self.v = np.asarray(IC)
                
    def get_m0(self, set_m0, soft = None):
        """
        Get the initial values for velocity either set directly by the user or for the cosmos == True case (explain what this means)
        """  
        if self.cosmos == False:
            self.m = np.array(set_m0.copy()).T
        else:
            #Relocate particles onto the nearest gridpoint
            self.x = np.rint(self.x).astype('int') % self.size[0]
            #Find the power spectrum
            k_x = np.real(np.fft.fft(self.x[:,0]))
            k_y = np.real(np.fft.fft(self.x[:,1]))
            k = np.sqrt(k_x**2 + k_y**2)
            #Apply softener to avoid blowup
            k[k<soft] = soft
            #Find effective mass
            m0 = set_m0/k**3
            
            self.m = np.array([m0.copy()]).T
                
                
                
        