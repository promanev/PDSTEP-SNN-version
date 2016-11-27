import numpy as np
#import math as m
#import os
#import sys
# VACC only: 
# import subprocess as subp
import random as rd
#import time as t
#import shutil, errno
import pylab as plb


#### ========== SUPPORT FCNs ============ ####

def readTxt(textFileName,data_type):
    # new better txt loader, outputs a 1D list of values in the file
    dataVar = []
    fid = open(textFileName,'r')
    for line in fid:
        split_line = line.split(' ')
        #print split_line
        for values in split_line:
            #print values
            if values!='\n':
                if data_type=='int':
                    dataVar += [int(values)]
                elif data_type=='float':
                    dataVar += [float(values)]
                # end IF
            # end IF
        # end FOR
    # end FOR                  
    fid.close()
    return dataVar
# end READTXT  
    
    
    
def plotSpikes(firings, neuron_signs):
    # read the firings file
    #
    firings = readTxt("firings1.txt", "int")
    #    
    # select times and ids:
    neuron_ids = firings[1::2]
    spike_times = firings[0::2]
    # get max neural sim time in ms
    max_t = max(spike_times)+1
    # get sample number:
    max_samp = len(spike_times)
    # init EEG array:
    EEG = np.zeros((max_t))
    # get number of neurons:
    # N = max(neuron_ids)
    
    # sum all of the spikes to get the pseudo-EEG:
    for samp in xrange(1,max_samp):
        # Adjust for MATLAB indexing by 1:
        curr_neur_num = neuron_ids[samp]
        # curr_neur_sign = (pop_member[curr_neur_num * row_len]-0.5)/0.5
        curr_neur_sign = neuron_signs[curr_neur_num]
        curr_spike_time = spike_times[samp]
        EEG[curr_spike_time] = EEG[curr_spike_time] + curr_neur_sign
    # end FOR
        
    # PERFORM SPECTRAL ANALYSIS:

    # split long recordings into 1000 ms epochs:
    epoch_size = 1000
    num_epochs = max_t / epoch_size
    
    for epoch in xrange(0, num_epochs):
        #    
        epoch_begin = epoch * epoch_size
        epoch_end = epoch_begin + epoch_size
        plb.figure(figsize=(24.0,20.0))
        plb.suptitle('Neural dynamics '+str(epoch_begin)+' to '+str(epoch_end)+' ms')
        time_array = (np.arange(max_t)+1)
        # Spike raster:
        plb.subplot(411)
        plb.title('Spikes')
        plb.xlabel('Time, ms')
        plb.ylabel('Neuron #')
        # find indices of spike_times that are closest to the epoch boundaries:
        time_begin_id,_  = min(enumerate(spike_times), key=lambda x: abs(x[1]-epoch_begin))
        time_end_id,_  = min(enumerate(spike_times), key=lambda x: abs(x[1]-epoch_end))
        # plot dots:
        plb.plot(spike_times[time_begin_id:time_end_id], neuron_ids[time_begin_id:time_end_id],'k.')
        
        # EEG:    
        plb.subplot(412)
        plb.title('Pseudo-EEG')
        plb.plot(time_array[epoch_begin:epoch_end],EEG[epoch_begin:epoch_end])
        plb.axis('tight')
        plb.xlabel('Time, ms')
        plb.ylabel('Spike count')
        
        # Periodogram:
        plb.subplot(413)
        plb.title('Spectral density')
        plb.psd(EEG[epoch_begin:epoch_end], Fs=1000)
        axis_vals = plb.axis()
        plb.axis([0, 100, axis_vals[2], axis_vals[3]])
        
        # Spectrogram:
        plb.subplot(414)
        plb.title('Spectrogram')
        plb.specgram(EEG[epoch_begin:epoch_end], Fs=1000)
        axis_vals = plb.axis()
        plb.axis([0, 1, 0, 100])
        plb.colorbar()
        plb.show()
        plb.savefig('neural_dynamics'+str(epoch)+'.png')        
# end plotSpikes

#### ========== MAIN SCRIPT =========== ####
#
# Number of neurons
N = 100 
# Length of recording in ms:
max_t = 1000
# list to keep firings in the same form as in Izhikevich 2003:
firings = []
# list to keep track of refractory periods:
refract = np.zeros((N))
# desired frequency:
fq = 25
period = 1000/25
# refractory period in ms:
refract_value = 10
# noise speard in ms:
spread = 3

# MAIN CYCLE through time:
for t in xrange(0,max_t):
    threshold = float(float(t%period)/period)
    # SECOND CYCLE through neurons:
    for n in xrange(0,N):
        #
        if rd.random() < threshold:
            # record spike and make neuron refractory            
            if refract[n]==0.0:
                refract[n]=refract_value
                firings.append(t+rd.randint(-spread,spread))
                firings.append(n)
            # end IF
        # end IF
        #
        # check if there are refractory periods, and decrease them:
        if refract[n]>0.0:
            refract[n]=refract[n]-1.0
        # end IF
    #end FOR N
# end FOR T
            
# Record firings into a .txt file:
fid = open("firings1.txt",'w+')
for item in xrange(0,len(firings)):
    if item == len(firings)-1:
        print item
        fid.write(str(firings[item]))
    else:    
        fid.write(str(firings[item])+" ")  
fid.close()
          
pos_signs = np.ones((N*0.8))
neg_signs = (-1)*np.ones((N*0.2))
neuron_signs = np.concatenate((pos_signs,neg_signs),axis=0)
plotSpikes(firings, neuron_signs)
            
#print "Step t=",t,"Threshold =",threshold    