#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:04:38 2020

@author: noahlefrancois
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py
#import glob
import json
from scipy.ndimage import gaussian_filter

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    #dqInfo = dataFile['quality']['simple']
    #qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'][()]
    utc=meta['UTCstart'][()]
    duration=meta['Duration'][()]
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

#Load the data for the 4 events in the provided data file
fnjson = "BBH_events_v3.json"
path = ""
events = json.load(open(path+fnjson,"r"))

i=0
name=[]
tevent = []
num_events = 4
NP = 131072
strain_h = np.zeros((num_events,NP))
strain_l = np.zeros((num_events,NP))
th = np.zeros((num_events,NP))
tl = np.zeros((num_events,NP))

for eventname in events:
    event = events[eventname]
    name.append(event['name'])
    fn_H1 = event['fn_H1'] 
    fn_L1 = event['fn_L1']  
    fn_template = event['fn_template'] 
    fs = event['fs']              
    tevent.append(event['tevent'])
    strain_h[i,:],dt,utc=read_file(path+fn_H1)
    strain_l[i,:],dt,utc=read_file(path+fn_L1)
    th[i,:],tl[i,:]=read_template(path+fn_template)
    i=i+1

#Suppress the output plots if False
outputPlots = True

#Parts a),b),c) combined to reduce # of for loops required
#Hanford model
strainWindowed_h = np.zeros((num_events,NP))
noiseModel_h = np.zeros((num_events,int(NP/2)+1))
window_th = np.zeros((num_events,NP))
matchedFilter_h = np.zeros((num_events,NP))
SNR_h = np.zeros((num_events,NP))
SNRPeak_h = np.zeros(num_events)
peakInd_h = np.zeros(num_events)

time = dt*np.arange(NP)

for i in range(num_events):
    NP = len(strain_h[i,:])
    xp = np.arange(NP)
    xp=xp-1.0*xp.mean()
    #Define the window function window(x) = 0.5+0.5cos(pi*x/L) 
    #This is maximum in the middle and goes to zero at the edges
    window_func = 0.5+0.5*np.cos(xp*np.pi/np.max(xp))
    #Multiply both the strain and template function by the window func and normalize
    strainWindowed_h[i,:] = strain_h[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    window_th[i,:] = th[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    
    #A) Define the noise model by taking the |FT|^2 and smoothing it. To do this, I assumed that 
    #the noise is uncorrelated and therefore used a gaussian filter as explained in ligo_slides.pdf
    tmp = np.fft.rfft(strainWindowed_h[i,:])*np.conj(np.fft.rfft(strainWindowed_h[i,:]))
    noiseModel_h[i,:] = gaussian_filter(np.abs(tmp),4)
    
    #B) Perform a matched filter search of the event using the provided template
    strainFT = np.fft.rfft(strainWindowed_h[i,:])
    templateFT = np.fft.rfft(window_th[i,:])
    #Whiten the power spectrums by dividing by sqrt(noiseModel)
    whiten_template = templateFT/np.sqrt(noiseModel_h[i,:])
    whiten_strain = strainFT/np.sqrt(noiseModel_h[i,:])
    matchedFilter_h[i,:] = np.fft.fftshift(np.fft.irfft(np.conj(whiten_template)*whiten_strain))
    
    #Get the index of the peak in the matched filter output
    timePeak_ind = np.argmax(np.abs(matchedFilter_h[i,:]))
    timePeak = time[timePeak_ind]
    if outputPlots:
        #Plot the matched filter results of the event
        plt.figure()
        plt.plot(time-timePeak, matchedFilter_h[i,:])
        plt.xlabel('Time elapsed since ' + str(tevent[i]))
        plt.ylabel('Matched Filter Output')
        plt.title('Hanford Event ' + name[i])
    
    #C) Calculate the signal to noise ratio of the event
    SNR_h[i,:] = matchedFilter_h[i,:]*(np.fft.irfft(np.conj(whiten_template))*np.fft.irfft(whiten_template))
    #Get the index of the peak in the SNR
    timeSNRPeak_ind = np.argmax(np.abs(SNR_h[i,:]))
    peakInd_h[i] = timeSNRPeak_ind
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak_h[i] = np.abs(SNR_h[i,timeSNRPeak_ind])
    print('Hanford Event ' + name[i] + ' SNR = ' + str(SNRPeak_h[i]))

"""
Output:
Hanford Event GW150914 SNR = 3.8717314222698023
Hanford Event LVT151012 SNR = 7.1265800638648376
Hanford Event GW151226 SNR = 0.35702943231737255
Hanford Event GW170104 SNR = 0.9341851030838584
"""
#Livingston model
strainWindowed_l = np.zeros((num_events,NP))
noiseModel_l = np.zeros((num_events,int(NP/2)+1))
window_tl = np.zeros((num_events,NP))
matchedFilter_l = np.zeros((num_events,NP))
SNR_l = np.zeros((num_events,NP))
SNRPeak_l = np.zeros(num_events)
peakInd_l = np.zeros(num_events)

for i in range(num_events):
    NP = len(strain_l[i,:])
    xp = np.arange(NP)
    xp=xp-1.0*xp.mean()
    #Define the window function window(x) = 0.5+0.5cos(pi*x/L) 
    #This is maximum in the middle and goes to zero at the edges
    window_func = 0.5+0.5*np.cos(xp*np.pi/np.max(xp))
    #Multiply both the strain and template function by the window func and normalize
    strainWindowed_l[i,:] = strain_l[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    window_tl[i,:] = tl[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    
   #A) Define the noise model by taking the |FT|^2 and smoothing it. To do this, I assumed that 
    #the noise is uncorrelated and therefore used a gaussian filter as explained in ligo_slides.pdf
    tmp = np.fft.rfft(strainWindowed_l[i,:])*np.conj(np.fft.rfft(strainWindowed_l[i,:]))
    noiseModel_l[i,:] = gaussian_filter(np.abs(tmp),4)
    
    #B) Perform a matched filter search of the event using the provided template
    strainFT = np.fft.rfft(strainWindowed_l[i,:])
    templateFT = np.fft.rfft(window_tl[i,:])
    #Whiten the power spectrums by dividing by sqrt(noiseModel)
    whiten_template = templateFT/np.sqrt(noiseModel_l[i,:])
    whiten_strain = strainFT/np.sqrt(noiseModel_l[i,:])
    matchedFilter_l[i,:] = np.fft.fftshift(np.fft.irfft(np.conj(whiten_template)*whiten_strain))
    
    #Get the index of the peak in the matched filter output
    timePeak_ind = np.argmax(np.abs(matchedFilter_l[i,:]))
    timePeak = time[timePeak_ind]
    if outputPlots:
        #Plot the matched filter results of the event
        plt.figure()
        plt.plot(time-timePeak, matchedFilter_l[i,:])
        plt.xlabel('Time elapsed since ' + str(tevent[i]))
        plt.ylabel('Matched Filter Output')
        plt.title('Livingston Event ' + name[i])
        
    #C) Calculate the signal to noise ratio of the event
    SNR_l[i,:] = matchedFilter_l[i,:]*(np.fft.irfft(np.conj(whiten_template))*np.fft.irfft(whiten_template))
    #Get the index of the peak in the SNR
    timeSNRPeak_ind = np.argmax(np.abs(SNR_l[i,:]))
    peakInd_l[i] = timeSNRPeak_ind
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak_l[i] = np.abs(SNR_l[i,timeSNRPeak_ind])
    print('Livingston Event ' + name[i] + ' SNR = ' + str(SNRPeak_l[i]))

"""
Output:
Livingston Event GW150914 SNR = 0.8969320015056981
Livingston Event LVT151012 SNR = 5.269947446743081
Livingston Event GW151226 SNR = 0.3886818863511979
Livingston Event GW170104 SNR = 0.8276815740714171 
"""

#C) Calculate the signal to noise ratio for the event combining the two measurements
SNRPeak_combined = np.zeros(num_events)
for i in range(num_events):
    #Get the combined magnitude of the SNR between the two events
    SNR_combined = np.sqrt(SNR_l[i,:]**2 + SNR_h[i,:]**2)
    #Get the index of the peak in the combined SNR
    timeSNRPeak_ind = np.argmax(SNR_combined)
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak_combined[i] = SNR_combined[timeSNRPeak_ind]
    print('Combined Event ' + name[i] + ' SNR = ' + str(SNRPeak_combined[i]))

"""
Output:
Combined Event GW150914 SNR = 3.8726613349224177
Combined Event LVT151012 SNR = 7.126583080163526
Combined Event GW151226 SNR = 0.39294296141353136
Combined Event GW170104 SNR = 0.9342501207119918
"""

#D) Compare SNR from the matched filter scatter to the SNR we got from our noise model
SNR_mfScatter_h = np.zeros(num_events)
SNR_mfScatter_l = np.zeros(num_events)
for i in range(num_events):
    #Estimate the SNR by taking the maximum signal magnitude divided by the scatter (ie standard deviation) in the signal
    SNR_mfScatter_h[i] = np.max(matchedFilter_h[i,:])/np.std(matchedFilter_h[i,:])
    print('Hanford Event ' + name[i] + '\nMatched Filter Scatter SNR = ' + str(SNR_mfScatter_h[i]) + '\nNoise Model SNR = ' + str(SNRPeak_h[i]))
    SNR_mfScatter_l[i] = np.max(matchedFilter_l[i,:])/np.std(matchedFilter_l[i,:])
    print('Livingston Event ' + name[i] + '\nMatched Filter Scatter SNR = ' + str(SNR_mfScatter_l[i]) + '\nNoise Model SNR = ' + str(SNRPeak_l[i]))

"""
This indicates that the matched filter scatter method overestimates the SNR compared to our
analytical noise model method.
Output:
Hanford Event GW150914
Matched Filter Scatter SNR = 13.488722414326844
Noise Model SNR = 3.8717314222698023
Livingston Event GW150914
Matched Filter Scatter SNR = 18.779726143669905
Noise Model SNR = 0.8969320015056078
Hanford Event LVT151012
Matched Filter Scatter SNR = 10.14981648661051
Noise Model SNR = 7.1265800638648376
Livingston Event LVT151012
Matched Filter Scatter SNR = 9.02538719741374
Noise Model SNR = 5.269947446743081
Hanford Event GW151226
Matched Filter Scatter SNR = 14.846555917152903
Noise Model SNR = 0.35702943231736706
Livingston Event GW151226
Matched Filter Scatter SNR = 7.207941955927423
Noise Model SNR = 0.3886818863511448
Hanford Event GW170104
Matched Filter Scatter SNR = 6.77389886440562
Noise Model SNR = 0.9341851030838584
Livingston Event GW170104
Matched Filter Scatter SNR = 9.55590230198362
Noise Model SNR = 0.8276815740714171
"""

#E) Find the frequency from each event where half the weight comes from above that frequency and half below
for i in range(num_events):
    #Hanford event
    #Get the whitened template for the event
    templateFT = np.fft.rfft(window_th[i,:])
    whiten_template = np.abs(templateFT/np.sqrt(noiseModel_h[i,:]))
    #Get the cumulative distribution function of whiten_template at each entry
    runningTotalWeight = np.cumsum(whiten_template)/np.sum(whiten_template)
    #Find the frequency where the CDF is closest to 1/2
    halfWeight_ind = np.argmin(np.abs(runningTotalWeight - 1/2))
    halfWeight_freq = np.fft.rfftfreq(NP, dt)[halfWeight_ind]
    print('Hanford Event ' + name[i] + '\nHalf-Weight Frequency = ' + str(halfWeight_freq) + ' Hz')
    #Livingston Event
    #Get the whitened template for the event
    templateFT = np.fft.rfft(window_tl[i,:])
    whiten_template = np.abs(templateFT/np.sqrt(noiseModel_l[i,:]))
    #Get the cumulative distribution function of whiten_template at each entry
    runningTotalWeight = np.cumsum(whiten_template)/np.sum(whiten_template)
    #Find the frequency where the CDF is closest to 1/2
    halfWeight_ind = np.argmin(np.abs(runningTotalWeight - 1/2))
    halfWeight_freq = np.fft.rfftfreq(NP, dt)[halfWeight_ind]
    print('Livingston Event ' + name[i] + '\nHalf-Weight Frequency = ' + str(halfWeight_freq) + ' Hz')


"""
Output:
Hanford Event GW150914
Half-Weight Frequency = 125.0 Hz
Livingston Event GW150914
Half-Weight Frequency = 133.78125 Hz
Hanford Event LVT151012
Half-Weight Frequency = 115.15625 Hz
Livingston Event LVT151012
Half-Weight Frequency = 129.375 Hz
Hanford Event GW151226
Half-Weight Frequency = 133.09375 Hz
Livingston Event GW151226
Half-Weight Frequency = 172.375 Hz
Hanford Event GW170104
Half-Weight Frequency = 125.09375 Hz
Livingston Event GW170104
Half-Weight Frequency = 113.125 Hz
"""

#F) How well can we localize the time of arrival, ie find the error on timePeak?
xErr_h = np.zeros(num_events)
xErr_l = np.zeros(num_events)
tErr_h = np.zeros(num_events)
tErr_l = np.zeros(num_events)
for i in range(num_events):
    #Find uncertainties for the Hanford event
    timeSNRPeak_ind = int(peakInd_h[i])
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak = SNRPeak_h[i]
    halfMax_SNR = SNRPeak/2
    halfMax_ind = np.argmin(np.abs(np.abs(SNR_h[i,:])-halfMax_SNR))
    #Time error is the full-width at half-maximum of the SNR peak
    tErr_h[i] = 2*np.abs(timeSNRPeak-time[halfMax_ind])
    #Spatial error is the time error times the speed of light (ie the speed of the signal)
    xErr_h[i] = tErr_h[i]*3*10**8
    print('Hanford Event ' + name[i] + '\nTime Uncertainty = ' + str(tErr_h[i]) + 's' + '\nPosition Uncertainty = ' + str(xErr_h[i]/10**3) + 'km')
    
    
    #Find uncertainties for the Livingston event
    timeSNRPeak_ind = int(peakInd_l[i])
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak = SNRPeak_l[i]
    halfMax_SNR = SNRPeak/2
    halfMax_ind = np.argmin(np.abs(np.abs(SNR_l[i,:])-halfMax_SNR))
    #Time error is the full-width at half-maximum of the SNR peak
    tErr_l[i] = 2*np.abs(timeSNRPeak-time[halfMax_ind])
    #Spatial error is the time error times the speed of light (ie the speed of the signal)
    xErr_l[i] = tErr_l[i]*3*10**8
    print('Livingston Event ' + name[i] + '\nTime Uncertainty = ' + str(tErr_l[i]) + 's' + '\nPosition Uncertainty = ' + str(xErr_l[i]/10**3) + 'km')
    print('Combined Positional Uncertainty = ' + str(xErr_h[i]/1000 + xErr_l[i]/1000) + 'km')
    
"""
Output:
Hanford Event GW150914
Time Uncertainty = 0.06982421875s
Position Uncertainty = 20947.265625km
Livingston Event GW150914
Time Uncertainty = 0.14501953125s
Position Uncertainty = 43505.859375km
Combined Positional Uncertainty = 64453.125km
Hanford Event LVT151012
Time Uncertainty = 0.0107421875s
Position Uncertainty = 3222.65625km
Livingston Event LVT151012
Time Uncertainty = 0.0009765625s
Position Uncertainty = 292.96875km
Combined Positional Uncertainty = 3515.625km
Hanford Event GW151226
Time Uncertainty = 0.00244140625s
Position Uncertainty = 732.421875km
Livingston Event GW151226
Time Uncertainty = 0.001953125s
Position Uncertainty = 585.9375km
Combined Positional Uncertainty = 1318.359375km
Hanford Event GW170104
Time Uncertainty = 0.0068359375s
Position Uncertainty = 2050.78125km
Livingston Event GW170104
Time Uncertainty = 0.13134765625s
Position Uncertainty = 39404.296875km
Combined Positional Uncertainty = 41455.078125km
"""
    
    
    
    