#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:06:35 2023

@author: thanos
"""

import scipy as sc 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal.windows import blackman






u = 346.15

a  = 0.2
b  = 0.2
c  = 0.195

Nval = 5

values = np.linspace(1,Nval,Nval,dtype=int)
flist = []
intlist = []


for l in values: #For loop to calculate the predicted resonant frequencies and their quantum numbers
    for m in values: 
        for n in values: 
            flist.append((u/2)*np.sqrt((l/a)**2 + (m/b)**2 + (n/c)**2))
            intlist.append([l,m,n])
            



Data = {'Integers ':intlist,'Predicted Frequencies ':flist} #Table that prints out the identified frequencies
mydata = pd.DataFrame(Data)
mydata2 = mydata.sort_values(by='Predicted Frequencies ')
print(mydata2)


wave = np.loadtxt("Data/BoxChuck1.txt") #Imported .txt audio file

N = len(wave) 
dur = wave[-1,0]
s_rate = N/dur #Sampling rate

amplitude = wave[:,1] #Wave amplitude list


w = blackman(N) #Blackman windowing function 

amplitudefft = np.abs(sc.fft.rfft(w*amplitude)) #scipy.rfft applied to the windowed amplitude list
xf = sc.fft.rfftfreq(N,1/s_rate) #frequency axis

res_freq_index = sc.signal.find_peaks(amplitudefft, prominence=(3))[0] #Peak finder algorithm 

freq_err = sc.signal.peak_widths(amplitudefft,res_freq_index,0.5)[0] #FWHM error on the frequency

freq_dict = {'Resonant Frequency': xf[res_freq_index],'Error':  freq_err} 
freq_dataframe = pd.DataFrame(freq_dict)
# print(freq_dataframe) #Uncomment to have the code print the resonant frequencies 


amplitude_plot = [] #Matching the found peak index to their frequency 
for i in range(len(res_freq_index)):
    amplitude_plot.append(xf[res_freq_index[i]])
    
    

            


rawchuck = np.loadtxt("Data/Chuck1Raw.txt") #Raw audio file of the wooden clicker 

Nr = len(rawchuck) #Same FFT code as above
durr = rawchuck[-1,0]
s_rater = Nr/durr

rawchuckamp = rawchuck[:,1]


wr = blackman(Nr)

rawchuckfft = 1.2*np.abs(sc.fft.rfft(wr*rawchuckamp))
xfchuck = sc.fft.rfftfreq(Nr,1/s_rater)



f, ax = plt.subplots(1,1) #FFT Plot
# ax.vlines(mydata2.get("Predicted Frequencies "), 0,200,colors='salmon',linewidth=0.8) #Uncomment for vertical lines showing the predicted resonant frequencies
   
# plt.plot(xfchuck,rawchuckfft,linestyle ='-',lw=1) #Uncomment for the raw wooden clicker audio over the fft plot 
ax.plot(xf, amplitudefft,color='dimgrey')
ax.scatter(amplitude_plot,amplitudefft[res_freq_index],s=10,edgecolors='r')


plt.grid(linestyle='--')
ax.set_xlim(0)
ax.set_ylim(0,np.max(amplitudefft)+np.std(amplitudefft))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Intensity')
ax.legend(['FFT','Wood Clicker Raw Sound (Scaled)','Identified Peaks'],fontsize = 7)



# plt.savefig('1o1_BoxChuck_TransformOverlay.png', dpi = 1080)

plt.tight_layout()
plt.minorticks_on()
plt.show()
