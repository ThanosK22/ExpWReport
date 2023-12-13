#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:28:13 2023

@author: thanos
"""

import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
from scipy.signal.windows import blackman


inputs = ['Data/Box1o1Sweep3k_4,5k.txt','Data/Box1o5Sweep3k_4,5k.txt'] #Two sweep audio files to floop through
legend = ['Sweep through the full Box','Sweep with the insert placed at 1/5 of the depth (scaled down)'] 
amp = [1,68/300] #Amplitude list to scale the second graph (for plotting reasons)
text = ['Width of first peak: ', 'Width of second peak: ']

f, ax = plt.subplots(1,1)
for j in range(len(inputs)): 

    wave = np.loadtxt(inputs[j])

    N = len(wave)
    dur = wave[-1,0]
    s_rate = N/dur

    amplitude = wave[:,1]
    w = blackman(N) #Window function 



    amplitudefft = amp[j]*np.abs(sc.fft.rfft(w*amplitude)) #fft 
    
    res_freq_index = sc.signal.find_peaks(amplitudefft, prominence=(20))[0] #Peak identification

    freq_err = sc.signal.peak_widths(amplitudefft,res_freq_index,0.5)[0] #Width of peak identification
    
    xf = sc.fft.rfftfreq(N,1/s_rate)

    
    ax.plot(xf, amplitudefft)
    
    print(text[j],'%.2f' % np.max(freq_err),'Hz')


    ax.grid(linestyle='--')
    # plt.xlim(0,xf[-1])
    ax.set_xlim(3000,4500)
    ax.set_ylim(0,np.max(amplitudefft)+3.5*np.std(amplitudefft))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Intensity')
    plt.legend(legend,fontsize = 7)
    
    




plt.tight_layout()
plt.minorticks_on()
# plt.savefig('sweep1o1fft',dpi=300)
plt.show()
    

    