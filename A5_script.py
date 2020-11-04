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
strain_h = np.zeros((num_events,131072))
strain_l = np.zeros((num_events,131072))
th = np.zeros((num_events,131072))
tl = np.zeros((num_events,131072))

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
outputPlots = False

#Parts a),b),c) combined to reduce # of for loops required
#Hanford model
strainWindowed_h = np.zeros((num_events,131072))
noiseModel_h = np.zeros((num_events,int(131072/2)+1))
window_th = np.zeros((num_events,131072))
matchedFilter_h = np.zeros((num_events,131072))
SNR_h = np.zeros((num_events,131072))
SNRPeak_h = np.zeros(num_events)

time = dt*np.arange(NP)

for i in range(num_events):
    NP = len(strain_h[i,:])
    xp = np.arange(NP)
    xp=xp-1.0*xp.mean()
    #Define the window function window(x) = 0.5+0.5cos(pi*x/L)
    window_func = 0.5+0.5*np.cos(xp*np.pi/np.max(xp))
    #Multiply both the strain and template function by the window func and normalize
    strainWindowed_h[i,:] = strain_h[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    window_th[i,:] = th[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    
    #A) Define the noise model (need to explain it. Gaussian filter for smoothing?)
    tmp = np.fft.rfft(strainWindowed_h[i,:])*np.conj(np.fft.rfft(strainWindowed_h[i,:]))
    noiseModel_h[i,:] = gaussian_filter(np.abs(tmp),4)
    
    #B) Perform a matched filter search of the event using the provided template
    strainFT = np.fft.rfft(strainWindowed_h[i,:])
    templateFT = np.fft.rfft(window_th[i,:])
    #explain whitening by dividing by sqrt(noiseModel)
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
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak_h[i] = np.abs(SNR_h[i,timeSNRPeak_ind])
    print('Hanford Event ' + name[i] + ' SNR = ' + str(SNRPeak_h[i]))

#Livingston model
strainWindowed_l = np.zeros((num_events,131072))
noiseModel_l = np.zeros((num_events,int(131072/2)+1))
window_tl = np.zeros((num_events,131072))
matchedFilter_l = np.zeros((num_events,131072))
SNR_l = np.zeros((num_events,131072))
SNRPeak_l = np.zeros(num_events)

for i in range(num_events):
    NP = len(strain_l[i,:])
    xp = np.arange(NP)
    xp=xp-1.0*xp.mean()
    #Define the window function window(x) = 0.5+0.5cos(pi*x/L)
    window_func = 0.5+0.5*np.cos(xp*np.pi/np.max(xp))
    #Multiply both the strain and template function by the window func and normalize
    strainWindowed_l[i,:] = strain_l[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    window_tl[i,:] = tl[i,:]*window_func/np.sqrt(np.mean(window_func**2))
    
    #A) Define the noise model (need to explain it. Gaussian filter for smoothing?)
    tmp = np.fft.rfft(strainWindowed_l[i,:])*np.conj(np.fft.rfft(strainWindowed_l[i,:]))
    noiseModel_l[i,:] = gaussian_filter(np.abs(tmp),4)
    
    #B) Perform a matched filter search of the event using the provided template
    strainFT = np.fft.rfft(strainWindowed_l[i,:])
    templateFT = np.fft.rfft(window_tl[i,:])
    #explain whitening by dividing by sqrt(noiseModel)
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
    timeSNRPeak = time[timeSNRPeak_ind]
    SNRPeak_l[i] = np.abs(SNR_l[i,timeSNRPeak_ind])
    print('Livingston Event ' + name[i] + ' SNR = ' + str(SNRPeak_l[i]))

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

#D) Compare SNR from the matched filter scatter to the SNR we got from our noise model
SNR_mfScatter_h = np.zeros(num_events)
SNR_mfScatter_l = np.zeros(num_events)
for i in range(num_events):
    #Not 100% sure if this is the correct way to estimate SNR using scatter but makes sense
    SNR_mfScatter_h[i] = np.max(matchedFilter_h[i,:])/np.std(matchedFilter_h[i,:])
    print('Hanford Event ' + name[i] + '\nMatched Filter Scatter SNR = ' + str(SNR_mfScatter_h[i]) + '\nNoise Model SNR = ' + str(SNRPeak_h[i]))
    SNR_mfScatter_l[i] = np.max(matchedFilter_l[i,:])/np.std(matchedFilter_l[i,:])
    print('Livingston Event ' + name[i] + '\nMatched Filter Scatter SNR = ' + str(SNR_mfScatter_l[i]) + '\nNoise Model SNR = ' + str(SNRPeak_l[i]))

"""
This indicates that the matched filter scatter method overestimates the SNR compared to our
analytical noise model method. [explain what this indicates]
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
    
    
    
    
    
    