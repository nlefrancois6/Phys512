N-Body Final Project
PHYS 512 Fall 2020
Noah LeFrancois

General Comments:
The N-Body code was implemented in both 2D and 3D; the majority of the 2D code was easily extended to 3D, however a few more significant changes needed to be made. This includes plotting the density field in 2D vs directly plotting all of the particles in a scatter plot in 3D; using the fast_histogram package to accelerate the 2D code but not being able to use it in 3D since the package has not been extended to 3D; and adaptation of the boundary conditions in the non-periodic case where the setting of Dirichlet boundary conditions requires some more careful handling of array indices.

I have generally used first-order schemes and derivatives, however I commented in a few places where I would have liked to implement a higher-order scheme if I had more time to continue the project (such as using a 2nd-order central differencing instead of 1st-order central differencing, which would have required some more complex handling of boundary conditions). This could potentially improve both the accuracy/energy conservation as well as increasing the time-step size allowed by the CFL condition.

As noted in the comments of Q4.py, I tested a few different implementations of FFT and histograms to see which one performed best; although this was not a very in-depth testing method due to time and computing resource constraints, I was able to achieve a slight improvement in speed by replacing the numpy.FFT library with the scipy.FFT one, as well as replacing the numpy.histogramdd function with the fast_histogram package (this chagne was only applied in 2D as mentioned above).

Discussion of Suggested Problems:
Part 1:
As seen in Q1_energy.png and Q1.gif, a single particle starting at rest remains motionless. 
This test case video was performed in 2D since the scatter3D axis limits don't display the domain well and I was able to produce a much clearer demonstration using the 2D colormap figure; however the dynamics are the same in both 2D and 3D.

Part 2:
As seen in Q2_energy.png and Q2.gif, the 2-particle orbit test case remained in its orbit over the course of multiple rotations. The energy spiked once every period as the particles passed closer and some blow-up error occured, however this spike soon returned back to approximately the original level. 
This test case video was performed in 2D since the scatter3D axis limits don't display the domain well & change with each frame, and I was able to produce a much clearer demonstration using the 2D colormap figure; however the dynamics are the same in both 2D and 3D.

Part 3:
The 3D periodic case video could not be uploaded to github as the file was too large due to the higher frame rate/smaller time step I used. I instead uploaded it to a google drive folder, accessible at the following link:
https://drive.google.com/file/d/1Qc3lViOMnDn2-G3OT5rqhVVu1s772amS/view?usp=sharing

In both BC cases, clusters form and eventually merge to form larger clusters. This collapse is much faster in the non-periodic case since there is no gravitational force pulling the cloud outwards as there is in the periodic case. This behaviour occurs in both 2D and 3D, however the visualization is not as clear for the 3D non-periodic case as the axis limits shift at each frame and as a result the reference frame is not at rest causing some distortion of the dynamics.

Energy was conserved better in the periodic case, with the 2D periodic case being the only one of the four cases not showing unconstrained growth of the energy over long times. I believe the poor energy conservation in the other cases (especially in 3D) is due in part to the time steps I needed to use in order to simulate an adequate time span for the dynamics without exceeding the limited memory space and speed of my laptop; using smaller time steps results in less rapid growth of the energy (see the smaller time steps in Q3_3D_P_energy.png and the above-linked video for the 3D periodic case). Running these simulations with smaller time steps (~1 or ~0.1 instead of ~10) would likely improve the energy conservation. 
This explanation is supported by the derivations done in the final exam question 3b), where I found that we needed dt << (soft)^3/2 * sqrt(5/6GM^2). This term is approximately equal to 0.2 for the parameters used in my simulations (soft = 0.8, G = 10). Unfortunately, running my simulation with a time step smaller than 0.2 and saving the video frames was too slow & costly for my memory space so I was not able to produce videos with this parameter. I was able to run the simulation without saving the frames and produced the energy plot 'Q3_energy_P_soft0p8_h0p1.png' which [demonstrates significantly better conservation of energy than the same case with larger timestep displayed in 'Q3_energy_P_soft0p8_longT.png'].

Additional improvements could be made by implementing some of the higher-order schemes I suggested in the General Comments section, in order to improve the accuracy and stability of the solver.

Part 4:
Although I believe I was able to implement this part successfully, I experienced significantly slower run-time and repeatedly reached my laptop's memory limits when trying to run it in both 2D and 3D. I was able to produce a video & energy plot of each, however the number of particles I was able to use in the 2D case was smaller than I would like to test. From these limited test cases (especially the 3D case), I observed that the cluster formation observed in Part 3 occured on much shorter time scales in this case, as these initial conditions already start with some clustering.

