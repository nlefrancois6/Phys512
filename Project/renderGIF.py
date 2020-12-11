#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:57:23 2020

@author: noahlefrancois
"""


import imageio
images = []

nsteps = 50

for t in range(nsteps):
    filename = 'Frames/Q3_frame'+str(t+1)+'.png'
    images.append(imageio.imread(filename))
    
imageio.mimsave('Videos/Q3.gif', images, duration=0.3)
