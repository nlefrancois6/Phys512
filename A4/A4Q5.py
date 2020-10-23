#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:26 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Part c) Pick a non-integer value of k and plot your analytic estimate of the DFT.
Show that FFT agrees (to within machine precision) with your analytic estimate.
Normally we think of the FFT of a pure sine wave to be a delta function. Are we close to that?
"""
#Calculate our analytic estimate of the DFT
def sinFT_analytic(x, k, N):
    kp = np.arange(N)
    terms = np.empty(N)
    for i in range(N):
        terms[i] = 1/(2j)*(-np.sum(np.exp(-2j*np.pi*(k-kp[i])*x/N) + np.exp(-2j*np.pi*(k+kp[i])*x/N)))
    terms = np.array(terms)
    return kp, terms

Np = 100
xp = np.arange(Np)
k = 7.5 #Non-integer value of k
f = np.sin(2*np.pi*k*xp/Np) #Define our sine wave

#Calculate our analytic DFT and the numpy fft of our sine wave, and plot the results
kp, f_An = sinFT_analytic(xp, k, Np)
f_Disc = np.fft.fft(f)

plt.figure()
plt.plot(xp, f)
plt.title('Sine Function')

plt.figure()
plt.plot(kp, abs(f_An), label='Analytic')
plt.plot(kp, abs(f_Disc), ':',label='FFT')
plt.legend()
plt.title('Fourier Transform of Sine Function')

plt.figure()
plt.plot(kp, abs(f_Disc)-abs(f_An),'*')
plt.title('Residuals')

print('The standard deviation between analytic and FFT values is ', f'{np.std(abs(f_Disc)-abs(f_An)):.4}')
"""
Output: The standard deviation between analytic and FFT values is  6.021e-14
The two functions agree very closely to machine precision, which is 1e-16.

The output is very close to a delta function at k and N-k, however it is not exactly a sharp/discontinuous
spike and has a bit of spectral leakage causing a widening of each peak. See A4Q5c_plot_FT and A4Q5c_plot_Residuals
"""

"""
Part d) Show that when we multiply by the window function 0.5-0.5cos(2pix/N), the spectral leakage for a
non-integer period sine wave drops dramatically
"""
#Define the window function
def window_func(x, N):
    window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
    return window
#Multiply sine wave by window function and take the FFT
window = window_func(xp, Np)
f_Disc_window = np.fft.fft(f*window)

plt.figure()
plt.plot(kp, abs(f_Disc), label='No window')
plt.plot(kp, abs(f_Disc_window), label='With window')
plt.legend()
plt.yscale('log')
plt.title('Fourier Transform of Sine Function')

"""
We can see from the plot (A4Q5d_plot_FT) that the FT after multiplying by the window function has a 
narrower peak at both k and N-k than the FT without the window function. The tradeoff is that the
magnitude of the peak is reduced when we multiply by the window function.
"""

"""
Part e) Show that the FT of the window is [N/2 N/4 0 0 ... 0 N/4].
Use this to show that you can get the windowed FT by appropriate combos of each point in the 
unwindowed FT and its immediate neighbours.
"""

window_FT = np.fft.fft(window)
given_windowFT = np.zeros(Np)
given_windowFT[0] = Np/2
given_windowFT[1] = -Np/4
given_windowFT[-1] = -Np/4

print('The standard deviation between given and FFT values is ', f'{np.std(abs(window_FT)-abs(given_windowFT)):.4}')

plt.figure()
plt.plot(kp, window_FT, label='FFT of Window')
plt.plot(kp, given_windowFT, ':', label = 'Given FT of Window')
plt.legend()
plt.title('Fourier Transform of Window Function')

"""
Output: The standard deviation between given and FFT values is  7.387e-16
See A4Q5e_plot_FTofWindow.png
When we flip the sign of the Np/4 entries, the FFT of the window agrees to machine precision with
the array we are given and comparing to.
This means the FFT of the window function is N/2 if k=0, -N/4 if k = +/-1, else 0. Since the multiplication 
of the window function by the sine wave is a convolution in Fourier space and the window function in
Fourier space is equal to the sum of these 3 delta functions at k=0, k=+/-1, this operation is just
going to select out 3 values for a given k. All other entries will be zero
"""

def windowed_FFT(F, N):
    terms = np.empty(N)
    for i in range(N):
        terms[i] = F[i]/2 - F[(i-1)%N]/4 - F[(i+1)%N]/4
    return terms

f_Disc_FixedWindow = windowed_FFT(f_Disc, Np)
print('The standard deviation between given and FFT values is ', f'{np.std(abs(f_Disc_window)-abs(f_Disc_FixedWindow)):.4}')

plt.figure()
plt.plot(kp, abs(f_Disc), label='FT of Sine')
plt.plot(kp, abs(f_Disc_window), label='FT of Sine*Window')
plt.plot(kp, abs(f_Disc_FixedWindow), ':', label='FT of Sine*Window using delta functions')
plt.legend()
plt.yscale('log')
plt.title('Fourier Transform of Sine Function')

"""
Output: The standard deviation between given and FFT values is  8.325e-16
See A4Q5e_plot_windowedSineFT.png
This agrees to machine precision with the FT of Sine*Window from Part d). This agreement confirms
that the method given in this question is equivalent to taking the windowed FT.
"""

