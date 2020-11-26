#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:10:12 2020

@author: noahlefrancois
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


"""
See attached plots Q1_Cplanes.png, Q1_pythonNoPlanes.png for output screenshots.

The C output can be clearly seen to line up along a set of planes when rotated the right way.
I was not able to see any such non-random structure in the Numpy output.

I was not able to get the local machine test to work.
"""
pyRands = np.loadtxt('rand_points_py.txt').T

cRands = np.loadtxt('rand_points_c.txt').T

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
# Creating plot
ax.scatter3D(pyRands[:][0], pyRands[:][1], pyRands[:][2], color = "green")
plt.title("Python Random Variables")
# show plot
plt.show()


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
# Creating plot
ax.scatter3D(cRands[0], cRands[1], cRands[2], color = "green")
plt.title("C Random Variables")
# show plot
plt.show()

