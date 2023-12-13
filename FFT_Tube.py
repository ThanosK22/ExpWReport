#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:35:43 2023

@author: thanos
"""

import scipy as sc 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal.windows import blackman

#This code performs a FFT on a single sound sample and plots the resonant frequencies as a function of peak index number. 

def func(x,a,b): #Function used later for the linear regression.
    return(a*x+b)

length = 0.403 #m, length of the tube 
lerr = 0.0005 #m, error in the length of the tube 

wave = np.loadtxt("Data/TubeChuck1.txt") #Imported .txt file with audio data from the microphome
peaklim = 10 #Maximum Peak index number to be used in plotting 


N = len(wave) #Amount of samples 
dur = wave[-1,0] #s, Length of sample
s_rate = N/dur #Sample rate 

amplitude = wave[:,1] #Amplitude column of the .tx file
w = blackman(N) #Window function 



amplitudefft = np.abs(sc.fft.rfft(w*amplitude)) #FFT on the windowed time domain sample
xf = sc.fft.rfftfreq(N,1/s_rate)

res_freq_index = sc.signal.find_peaks(amplitudefft, prominence=(2.5))[0] #Scipy peakfinng algorithm used to idenitfy the peaks in the intensity

freq_err = sc.signal.peak_widths(amplitudefft,res_freq_index,0.5)[0] #Error in the resonant frequency being approximated as the FWHM of the peak

freq_dict = {'Resonant Frequency': xf[res_freq_index],'Error':  freq_err} #Pandas dictionary, prints out the resonant frequencies
freq_dataframe = pd.DataFrame(freq_dict)
print(freq_dataframe)


amplitude_plot = [] #Mathces the identified indecies of the peaks to the frequency at that index 
for i in range(len(res_freq_index)):
    amplitude_plot.append(xf[res_freq_index[i]])
    
indices = np.linspace(1,len(amplitudefft[res_freq_index]),len(amplitudefft[res_freq_index])) #Creates a list out of all the indexes identified, used for plotting
    


popt,pcov = sc.optimize.curve_fit(func,indices[:peaklim],xf[res_freq_index[:peaklim]],sigma=1./((freq_err[:peaklim])*(freq_err[:peaklim]))) #Linear regression algorithm, optimises for a least squares fit with an error in y
perr = np.sqrt(np.diag(pcov))

print('———————————–')
print('fit parameter 1-sigma error') #Prints parameters a and b from the straight line function (func) defined at the start 
for i in range(len(popt)): 
    print(str(popt[i])+' +- '+str(perr[i]))

print('———————————–')
sp_sound=2*length*popt[0]
print('speed of sound: ','%.2f' %sp_sound,'m/s') #Converts the slope of the frequency - index curve to a speed of sound calculation, the error was calculated using the error propagation formula
print('error: +-','%.2f' %np.sqrt((2*popt[0]*lerr)**2 + (2*length*perr[0])**2),'m/s')

fit = func(indices[:peaklim],*popt) #Fit function for striaght line plotting 


f, (ax1,ax2) = plt.subplots(1,2) #Plots of the FFT and the Resonant Frequency - Index Number Curve. 
   
ax1.plot(xf, amplitudefft,'-')
ax1.scatter(amplitude_plot,amplitudefft[res_freq_index],s=10,edgecolors='r')


ax1.grid(linestyle='--')
ax1.set_xlim(0,xf[-1])
ax1.set_ylim(0,np.max(amplitudefft)+np.std(amplitudefft))
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Intensity')
ax1.legend(['FFT','Identified Peaks'],fontsize = 8)


ax2.scatter(indices[:peaklim],xf[res_freq_index[:peaklim]],s=1,color = 'r')
ax2.errorbar(indices[:peaklim],xf[res_freq_index[:peaklim]],yerr=freq_err[:peaklim],xerr=0,ecolor='k',fmt='none',label='data')
ax2.plot(indices[:peaklim],fit,'b',lw=1,label='Best fit line')
ax2.grid(linestyle='--')
ax2.set_xlim(0)
ax2.set_ylim(0)
ax2.set_xlabel('Peak Index')
ax2.set_ylabel('Resonant Frequency (Hz)')
ax2.legend()

plt.tight_layout()
plt.minorticks_on()
plt.show(ax1)
plt.show(ax2)