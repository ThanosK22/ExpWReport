#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:14:18 2023

@author: thanos
"""

import scipy as sc 
import numpy as np 
from scipy.signal.windows import blackman

#This code loops FFT_Tube through all the data collected and calculates the Mean value of the speed of sound with error

def func(x,a,b): #Function used later for the linear regression.
    return(a*x+b)

length = 0.403 #m, length of the tube 
lerr = 0.0005 #m, error in the length of the tube 

filelist = ['Data/TubeChuck1.txt','Data/TubeClick1edit.txt','Data/TubeClick2edit.txt','Data/TubeClick3edit.txt']
sp_sound_list = np.zeros(len(filelist))
sp_sound_err_list = np.zeros(len(filelist))

for i in range(len(filelist)):

    wave = np.loadtxt(filelist[i]) #Imported .txt file with audio data from the microphome
    peaklim = 8 #Maximum Peak index number to be used in plotting 
    
    
    N = len(wave) #Amount of samples 
    dur = wave[-1,0] #s, Length of sample
    s_rate = N/dur #Sample rate 
    
    amplitude = wave[:,1] #Amplitude column of the .tx file
    w = blackman(N) #Window function 
    
    
    
    amplitudefft = np.abs(sc.fft.rfft(w*amplitude)) #FFT on the windowed time domain sample
    xf = sc.fft.rfftfreq(N,1/s_rate)
    
    res_freq_index = sc.signal.find_peaks(amplitudefft, prominence=(2.5))[0] #Scipy peakfinng algorithm used to idenitfy the peaks in the intensity
    
    freq_err = sc.signal.peak_widths(amplitudefft,res_freq_index,0.5)[0] #Error in the resonant frequency being approximated as the FWHM of the peak
    
 
    
    
    amplitude_plot = [] #Matches the identified indecies of the peaks to the frequency at that index 
    for k in range(len(res_freq_index)):
        amplitude_plot.append(xf[res_freq_index[k]])
        
    indices = np.linspace(1,len(amplitudefft[res_freq_index]),len(amplitudefft[res_freq_index])) #Creates a list out of all the indexes identified, used for plotting
        
    
    
    popt,pcov = sc.optimize.curve_fit(func,indices[:peaklim],xf[res_freq_index[:peaklim]],sigma=1./((freq_err[:peaklim])*(freq_err[:peaklim]))) #Linear regression algorithm, optimises for a least squares fit with an error in y
    perr = np.sqrt(np.diag(pcov))
    
    
    
    sp_sound_list[i]=2*length*popt[0]
    sp_sound_err_list[i] = np.sqrt((2*popt[0]*lerr)**2 + (2*length*perr[0])**2)
    
    
    fit = func(indices[:peaklim],*popt) #Fit function for striaght line plotting 

sp_sound = np.average(sp_sound_list) #Average value of the speed of sound list 
sp_sound_err=0
a=0
for j in range(len(sp_sound_err_list)): #Error propagation loop to find the final error in the speed of sound 
    a += (sp_sound_err_list[j]**2)
sp_sound_err = (1/len(sp_sound_err_list))*np.sqrt(a)

print('Speed of sound: ','%.2f' % sp_sound, 'm/s')
print('Error : +-','%.2f' % sp_sound_err,'m/s')

    
    