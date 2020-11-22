#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:57:02 2020

@author: noahlefrancois
"""

import numpy as np

class NBody_solver:
    def __init__(self,size,particles,dt,soft=0.1,G=1,boundaryCondition='Periodic'):
        """
        The NBody class that specifies the simulation. 
        Input(s):
            - N (x,y): size of the grid. NOTE: Only accepts square grids currently
            - particles (system_init object): List of velocity&position for all particles
            - dt (float): step size for time-stepping 
            - soft (float): softening parameter for green function
            - G (float): Gravitational constant (default 1.0)
            - boundary type: Either set to Periodic or Non-Periodic
        """

        self.BC = boundaryCondition

        #For non-periodic conditions, the grid is padded to double the actual size 
        #with empty space to avoid wrap-around effects.
        if self.BC == 'Non-Periodic':
            self.size = (2*size[0],2*size[1])   
        else:
            self.size = size
        self.soft = soft
        self.G = G
        self.dt = dt
        self.x = particles.x
        self.v = particles.v
        self.m = particles.m
        x, y = np.linspace(0, self.size[0]-1, self.size[0]), np.linspace(0, self.size[1]-1, self.size[1])
        self.mesh = np.array(np.meshgrid(x,y))
        self.get_NGPdensity()
        self.get_greensFunc()
    
    def get_NGPdensity(self):
        """
        Get the density function rho on the grid using the Nearest Grid Points method, which approximates
        the density by lumping each particle's mass into the nearest grid point.
        """
       #Get nearest gridpoint index
        self.x_ind = np.rint(self.x).astype('int') % self.size[0]
        
        #Store the grid of particle indices to access later
        self.x_indexList = tuple(self.x_ind[:, i] for i in range(2))
        
        #Add up the contributions of all particles to get the total density at each grid point
        x_gridpoints = np.linspace(0, self.size[0]-1, self.size[0]+1)
        bin_coords = np.repeat([x_gridpoints], 2, axis=0)
        GP_densities = np.histogramdd(self.x_ind, bins = bin_coords, weights = self.m.flatten())
        self.rho = GP_densities[0]
        
    def get_greensFunc(self):
        """
        Get the Green's function of the particles on the grid
        Note: this will only work for square grids as there was not enough time 
        to implement this with rectangular grids. 
        """
        r = np.sum(self.mesh**2,axis=0)
        r[r<self.soft**2] = self.soft**2
        r += self.soft**2
        r = np.sqrt(r)
        
        g = 1/(4*np.pi*r)
        
        if self.BC == 'Periodic':
            #Handle the corner periodic behaviour
            h_x,h_y = self.size[0]//2, self.size[1]//2

            try:
                g[h_x:, :h_y] = np.flip(g[:h_x,:h_y],axis=0)
                g[:,h_y:] = np.flip(g[:,:h_y],axis=1)
            except: 
                g[h_x:, :h_y+1] = np.flip(g[:h_x+1,:h_y+1],axis=0)
                g[:,h_y:] = np.flip(g[:,:h_y+1],axis=1)
        self.green = g
        
    def get_Potential(self):
        """
        Get the potential function by solving the Laplace equation L(Psi) = rho. The simplest way to
        do this is to take the convolution of rho with the Green's function.
        If time, could replace this with a more efficient Laplace solver such as Cyclic Reduction
        """
        
        rho_FFT = np.fft.rfftn(self.rho)
        green_FFT = np.fft.rfftn(self.green)
        
        psi_FFT = rho_FFT*green_FFT
        psi = np.fft.irfftn(psi_FFT)

        #Shift psi back to the centre of the ptcls (not sure about this step)
        for i in range(2):
            psi = 0.5*(np.roll(psi,1,axis=i)+psi)
        
        #If the boundary conditions is not periodic, set the potential to 0
        #For non-periodic boundary conditions, use Dirichlet BC with Psi|_boundary = 0
        if self.BC == 'Non-Periodic':
            psi[0:,0] = 0
            psi[0,-1] = 0
            psi[-1:,-1] = 0
            psi[-1:,0] = 0 

        self.psi = psi
    
    def get_Forces(self):
        """
        The force at each point is defined as the gradient of the potential
        Evaluate the gradient using central differencing (could use a higher-order method if time)
        We then apply the force at each grid point to each particle lumped onto that grid point by the NGP method
        """
    
        F_gp = np.zeros([2,self.size[0],self.size[1]])
        self.get_Potential()
        #Central differencing to get force field on the grid points 
        #Could replace this with a higher order scheme
        #np.roll(x,n,axis=0) shifts the array entries of x by n indices along axis 0
        F_gp[0] = 0.5*(np.roll(self.psi,1,axis=0)-np.roll(self.psi,-1,axis=0))
        F_gp[1] = 0.5*(np.roll(self.psi,1,axis=1)-np.roll(self.psi,-1,axis=1))
        F_gp = -self.G*self.rho*F_gp
        #Apply the force to each particle
        F_ptcls = np.moveaxis(F_gp,0,-1)
        F = F_ptcls[self.x_indexList]
        self.F = F
    
    def get_totalEnergy(self):
        """
        Track the total energy of the system as the sum of the kinetic and potential energies
        """
        #print(len(self.v))
        #vTot = np.empty(len(self.v))
        #for p in range(len(self.v)):
         #   vTot[p] = np.sqrt(self.v[p][0]**2 + self.v[p][1]**2)
        vTot = np.sqrt(self.v[:,0]**2 + self.v[:,1]**2)
        #print(self.m.shape)
        KE = np.sum(self.m*vTot**2)
        #Would be nice to take psi & rho as inputs to avoid calculating them redundantly
        psi = self.psi
        rho = self.rho
        PE = -0.5*np.sum(np.sum(psi)*rho)
        TE = KE + PE
        self.E = TE
    
    def time_stepping_leapfrog(self):
        """
        Evolves the particle's position and momentum using the leap frog method seen in class
        I'd eventually like to replace the leap frog with some high-order method but this will do for now
        """
        #velocity update step
        self.v[:,0] = self.v[:,0]+self.F[:,0]*self.dt/self.m
        self.v[:,1] = self.v[:,1]+self.F[:,1]*self.dt/self.m
        #position update steps
        self.x = self.x+self.v*self.dt
        self.x = self.x%self.size[0]


    def advance_timeStep(self, file_save = None):
        #file_save=['x_data','E_data',1]
        """
        Take one step forward in time 
        Save and output the particle positions and the total energy after the step
        
        Input:
            file_save = (x_file, E_file, NParticles): the filenames to which we want to save 
            the position and energy data. NParticles is the number of particles to track
        """
        
        
        #Get the forces from the current particle distribution (psi is defined during this operation)
        self.get_Forces()
        #Get the total energy for the current particle distribution (needs psi)
        self.get_totalEnergy()
        energyTot = self.E
        #Move the particles according to the force field to get the new particle distribution
        #This method will update the positions and velocities of the particles
        self.time_stepping_leapfrog()
        #Update rho and Green's function for the new particle distribution
        self.get_NGPdensity()
        self.get_greensFunc()
        
        
        #Save the data
        """
        print('File save: ', file_save)
        if file_save is not 0:
            file_save[1].write(f'{energyTot}\n') #might be able to replace this w just self.E
            file_save[1].close()
            file_save[0].write(f'{self.x[:file_save[2]]}\n')
            file_save[0].close()
        """
                
        return self.E, self.x

