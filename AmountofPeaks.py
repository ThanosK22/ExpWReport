#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:26:48 2023

@author: thanos
"""

import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
from scipy.signal.windows import blackman

def func(x,a,b): #Function used later for the linear regression.
    return(a*x+b)


inputs = ['Data/BoxChuck1.txt','Data/1o2_Chuck1.txt','Data/1o3_Chuck1.txt','Data/1o4_Chuck1.txt','Data/1o5_Chuck1.txt','Data/0.034_WBoxChuck1.txt']

depth = np.array([0.2,0.1,0.2-0.066,0.2-0.05,0.2-0.04,0.2-0.034])

derr = 0.0005
derrlist=0.0005*np.ones(len(inputs))

rawchuck = np.loadtxt("Data/Chuck1Raw.txt")

Nr = len(rawchuck)
durr = rawchuck[-1,0]
s_rater = Nr/durr

rawchuckamp = rawchuck[:,1]

rawchuckfft = np.abs(sc.fft.rfft(rawchuckamp))
xfchuck = sc.fft.rfftfreq(Nr,1/s_rater)

NofPeaks = np.zeros(len(inputs))

for j in inputs: #Loops through the inputs and calculates the number of peaks for each one of them. 

    wave = np.loadtxt(j)

    N = len(wave)
    dur = wave[-1,0]
    s_rate = N/dur

    amplitude = wave[:,1]

    w = blackman(N)

    amplitudefft = np.abs(sc.fft.rfft(w*amplitude))
    xf = sc.fft.rfftfreq(N,1/s_rate)

    res_freq_index = sc.signal.find_peaks(amplitudefft, prominence=(2.5))[0]

    freq_err = sc.signal.peak_widths(amplitudefft,res_freq_index,0.5)[0]



    amplitude_plot = []
    for i in range(len(res_freq_index)):
        amplitude_plot.append(xf[res_freq_index[i]])
        
    indices = np.linspace(1,len(amplitudefft[res_freq_index]),len(amplitudefft[res_freq_index]))
        

        
    rawchuckfft = (np.max(amplitudefft)/np.max(rawchuckfft))*rawchuckfft
       
    
    NofPeaks[inputs.index(j)] = len(res_freq_index)+1
    
popt,pcov = sc.optimize.curve_fit(func,depth,NofPeaks,sigma=1./(derrlist*derrlist)) #Linear regression algorithm, optimises for a least squares fit with an error in y
perr = np.sqrt(np.diag(pcov))

print('———————————–')
print('fit parameter 1-sigma error') #Prints parameters a and b from the straight line function (func) defined at the start 
for i in range(len(popt)): 
    print(str(popt[i])+' +- '+str(perr[i]))


fit = func(depth,float(popt[0]),float(popt[1])) #Fit function for striaght line plotting 
    
slope, intercept, r_value, p_value, std_err = sc.stats.linregress(depth, NofPeaks) #Used for R^2
    
print('R square: ',r_value**2)

f,ax = plt.subplots(1,1)
ax.scatter(depth,NofPeaks)
plt.grid(linestyle='--')
plt.plot(depth,fit)
ax.errorbar(depth, NofPeaks, xerr=derrlist,yerr=0,ecolor='k',fmt='none')
ax.set_title('No. of peaks as a function of box depth')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Amount of Peaks')
plt.legend(['Data Points','Line of Best Fit'],fontsize=8)
plt.savefig('Report/NumberofPeaks.png',dpi=300)
plt.minorticks_on()
plt.show()

    