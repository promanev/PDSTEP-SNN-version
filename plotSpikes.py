import numpy as np
import pylab as plb

def readTxt(textFileName,data_type):
    # new better txt loader, outputs a 1D list of values in the file
    dataVar = []
    fid = open(textFileName,'r')
    for line in fid:
        split_line = line.split(' ')
        for values in split_line:
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


def plotSpikes(pop_member, row_len):
    # read the firings file
    firings = readTxt("firings.txt", "int")
    # select times and ids:
    neuron_ids = firings[1::2]
    spike_times = firings[0::2]
    # get max neural sim time in ms
    max_t = max(spike_times)
    # get sample number:
    max_samp = len(spike_times)
    # init EEG array:
    EEG = np.zeros((max_t))
    # get number of neurons:
    # N = max(neuron_ids)
    
    # sum all of the spikes to get the pseudo-EEG:
    for samp in xrange(1,max_samp):
        # Adjust for MATLAB indexing by 1:
        curr_neur_num = neuron_ids[samp]-1
        curr_neur_sign = (pop_member[curr_neur_num * row_len]-0.5)/0.5
        curr_spike_time = spike_times[samp]-1
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
        plb.savefig('neural_dynamics'+str(epoch)+'.png')        
# end plotSpikes
