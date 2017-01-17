# This script attempts to model dynamics of the cortico-basal-ganglia-thalamic
# loop with beta event-related desynchronization (ERD). The beta ERD is simulated 
# by making the BG network consisting of fast inhibitory and CLass 1/2 excitatory 
# neurons that tends to oscillate in the beta range but shifts into the gamma if 
# there is a stronger drive to its inhibitory neurons (~ 3 times more to inhibitory
# than to excitatory). 
# 
# Need to code up:
# 
# * Cortical network
# * Gate 
# * Fcn to trace beta power throughout the time of simulation, detect changes in 
#   beta power before/during/after the change of "motor states". 
# * Update plotSpikes so it can now work with new data representation

# ============= IMPORTS =============
import numpy as np
import math as m
import random as rd
import time as t
import shutil, errno
import pylab as plb


# STRUCTURE-like class:
# similar to C and MATLAB structures. Use as:
# Example: point = Bunch(datum=y, squared = y*y, coord x)
# point.isok = 1, etc.
#
class Bunch:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)
# end CLASS BUNCH        

# UTILITY FUNCTIONS:
    
def pop2wts(data, row_len):
    # Function makes wts matrice out of 1D pop.member
    wts = np.zeros((len(data)/row_len, row_len))
    # print "wts size =", len(wts)
    # print "Pop.member length =",len(data)    
    for j in xrange(0,len(data)):
        # print "Data item # ",j
        # print "Trying to assess wts[",m.ceil(j/row_len),"][",j - int(m.ceil(j/row_len))*row_len,"]"
        wts[int(m.ceil(j/row_len))][j - int(m.ceil(j/row_len))*row_len] = data[j]
    return wts    
# end POP2WTS
    
def writeWeights(fileName, data, row_len):
    fid = open(fileName,'w+')
    for j in xrange(0,len(data)):
        if m.fmod(j+1,row_len)==0:
            fid.write(str(data[j])+"\n")
        else:
            fid.write(str(data[j])+" ")
    # end FOR
    fid.close()
# end DEF    
    
def mean(numbers): # Arithmetic mean fcn
    return float(sum(numbers)) / max(len(numbers), 1)
# end MEAN fcn

def plotSpikes(pop_member, firings):
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
        
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
# end COPYANYTHING  
        
def genSpikes(tick, Fq, refract, N):
    # Fcn generates one-dim vector of inputs size from a simulated rhythm  
    #
    # vector to output:
    outputs = np.zeros((N))      
    period = 1000/Fq
    # refractory period in ms:
    refract_value = 10
    # noise speard in ms:
    spread = 3
    # the closer you are to this threshold the higher the probability of a spike:
    threshold = float(float(tick%period)/period)
    # 
    for n in xrange(0,N):
        #
        if rd.random() < threshold:
            # record spike and make neuron refractory            
            if refract[n]==0.0:
                refract[n]=refract_value
                outputs[n]=1
            # end IF
        # end IF
        #         
        # check if there are refractory periods, and decrease them:
        if refract[n]>0.0:
            refract[n]=refract[n]-1.0
        # end IF
    # end FOR N
    #
    return outputs, refract
# end GENSPIKES        

    
# ============= PRINCIPAL FUNCTIONS =============
    
def simSNN(pop_member, SNNparam, v, u, inputs):
    # Function that creates an SNN and runs it for 1 sim. step
    # pop_member - part of the population vector with weights pertaining to this network
    # SNNparam - structure with a range of network-related parameters
    # v, u - membrane potential and its recovery variable (a,b,c,d are included in SNNparam)
    # inputs - binary vector containing spike info of previous network's outputs
    # 
    # Begin:  
    #
    Ne = SNNparam.ne # number of excitatory hidden neurons
    Ni = SNNparam.ni # number of inhibitory hidden neurons
    N_hidden = Ne + Ni # total number of hidden neurons
    N_out = SNNparam.num_out # number of output neurons
    N_in = SNNparam.num_in # number of input neurons
    N = N_hidden + N_out # number of simulated neurons (only hidden and output neurons)
    #
    # based on the length of input vector decide if the gate or sm loop are simulated:
    #
    if len(inputs)>SNNparam.num_in: # only SM loop has inputs from both gate and its sensors
        is_smloop = 1
        N_aux = len(inputs) - SNNparam.num_in # number of additional, auxilary to this SNN, inputs
    else:
        is_smloop = 0
    # end IF
    # 
    # now create weight matrices:
    if is_smloop:
        gate_hidden_mark = N_aux * N_hidden
        sensor_hidden_mark = gate_hidden_mark + N_in*N_hidden
        hidden_hidden_mark = sensor_hidden_mark + N_hidden*N_hidden
        Wgh = pop2wts(pop_member[0:gate_hidden_mark], N_aux)
        Wsh = pop2wts(pop_member[gate_hidden_mark:sensor_hidden_mark], N_in)
        Whh = pop2wts(pop_member[sensor_hidden_mark:hidden_hidden_mark], N_hidden)
        # also break down inputs from gate and sm loop itself in two:
        aux_input_vals = inputs[0:N_aux]
        smloop_input = inputs[N_aux:]
    else:
        sensor_hidden_mark = N_in*N_hidden
        hidden_hidden_mark = sensor_hidden_mark + N_hidden*N_hidden
        Wsh = pop2wts(pop_member[0:sensor_hidden_mark], N_in)
        Whh = pop2wts(pop_member[sensor_hidden_mark:hidden_hidden_mark], N_in)
        
    # end IF    
    
    hidden_motor_mark = hidden_hidden_mark + N_out * N_hidden    
    Whm = pop2wts(pop_member[hidden_hidden_mark:hidden_motor_mark], N_out)

    # Izhikevich model's parameters:
    # Start with setting all of neurons to Class 1 Excitatory 
    # (the last N_out neurons - output neurons - will stay this kind):
    a = np.full((N, 1), SNNparam.a[0])
    b = np.full((N, 1), SNNparam.b[0])
    c = np.full((N, 1), SNNparam.c[0])
    d = np.full((N, 1), SNNparam.d[0])
    # v = np.full((N, 1), -65.0)
    # u = np.full((N, 1), v[0]*b[0])
    
    # Now replace first 0.3*N_hidden neurons with Class 2 (about 30% are perfect for beta/gamma dichotomy)
    ratio_neur_type = 0.3 # how many Class 2 neurons are added into the mix
    for idx in xrange(0,int(ratio_neur_type*N_hidden)):
        a[idx] = SNNparam.a[1]
        b[idx] = SNNparam.b[1]
        c[idx] = SNNparam.c[1]
        d[idx] = SNNparam.d[1]
        # need to update u[idx], since b[idx] changed:
        # u[idx] = v[idx] * b[idx]
    # Set the last 20% of hidden neurons to fast inhibitory params:
    for idx in xrange(int(0.8*N_hidden),N_hidden):
        a[idx] = SNNparam.a[2]
        b[idx] = SNNparam.b[2]
        c[idx] = SNNparam.c[2]
        d[idx] = SNNparam.d[2]
        # need to update u[idx], since b[idx] changed:
        # u[idx] = v[idx] * b[idx]
    #end FOR 
    #
    h = 2.0
    # create influence vector by feeding it sensor and gate data:
    I = np.zeros((N_hidden))
    # create influence vector for output neurons:
    I_out = np.zeros((N_out))
    #
    if is_smloop: # SMLOOP SIMULATED
        for i in xrange(0, N_hidden):
            # auxilary:
            for j in xrange(0, N_aux):
                if Wgh[i][j]>0:
                    if aux_input_vals[j]>0:
                        I[i] += 1 * SNNparam.ig # ig = input gain
            # sensory:
            for j in xrange(0, N_in):
                if Wsh[i][j]>0:
                    if smloop_input[j]>0:
                        I[i] += 1 * SNNparam.ig           
        # end FOR
    else: # GATE SIMULATED!!!
         for i in xrange(0, N_hidden):               
            for j in xrange(0, N_in):
                if Wsh[i][j]>0:
                    if inputs[j]>0:
                        if gate.cue_flag: # cue_flag is raised if a cue was introduced and for several ticks afterwards, this is calculated in getFitness()
                            if i < int(0.8 * (gate.ni + gate.ne)): # if this is an excitatory gate neuron
                                I[i] += 1 * SNNparam.ig
                            else: # if this is an INHIBITORY gate neuron (gets increased drive after the cue to cause beta ERD)
                                I[i] += 1 * SNNparam.ig * 5
                            # end IF    
                        else: # if we are outside of "motor change" (i.e., before or sufficiently later after the cue)
                            I[i] += 1 * SNNparam.ig
                        # end IF CUE_FLAG
                    # end IF INPUTS
                # end IF WTS
            # end FOR all inputs
         # end FOR all hidden
    # end IF GATE or SMLOOP
    #                             
    # Now go through all of the hidden neurons and detect if any of them spiked:        
    for idx in xrange(0, N_hidden):
        # if neuron spiked:
        if v[idx]>30:
            #
            # reset membrane potential of fired neuron:
            v[idx] = c[idx]
            u[idx] = u[idx] + d[idx];    
   
            # propagate influence of this spike by adjusting all of the I-s values that this neuron connects to  
            #    print "Fired idx =",idx
            #    print "Before I =",I
            for idx2 in xrange(0, N_hidden): # Here, it's assumed that each neuron is connected to each other (including self)
                if Whh[idx][idx2]>0:                
                    if idx > int(0.8*N_hidden):
                        I[idx2] = I[idx2] - 1.0 * SNNparam.hg # hg = hidden gain
                    else:
                        I[idx2] = I[idx2] + 1.0 * SNNparam.hg
                    # end IF
                    # print "AFTER I[", afferent_id,"]=",I[afferent_id]   
                # end IF  
            # end FOR
                        
            # similarly, adjust I_out:
            for idx2 in xrange(0, N_out):
                if Whm[idx][idx2]>0:
                    if idx > int(0.8*N_hidden):
                        I_out[idx2] = I_out[idx2] - 1.0 * SNNparam.og # og = output gain
                    else:
                        I_out[idx2] = I_out[idx2] + 1.0 * SNNparam.og
                # end IF  
            # end FOR                        
        
    # integrate numerically membrane potentials:
    for idx in xrange(0, N_hidden):
        for integ_tick in xrange(0, int(h)):
            # print "v[",idx,"]=",v[idx]
            # print "1/h = ",1/h
            # print "0.04*(v[",idx,"]**2)=",0.04 * (v[idx] ** 2)
            # print "5*v[",idx,"]=",5*v[idx]
            # print "140 - u[",idx,"] + I[",idx,"]=",(140 - u[idx] + I[idx])
            v[idx] += (1/h) * (0.04 * (v[idx] ** 2) + 5 * v[idx] + 140 - u[idx] + I[idx])
        # end FOR   
        #  
        # update u:
        u[idx] += a[idx] * (b[idx] * v[idx] - u[idx])
    # end FOR
            
    # now update output neurons
    # first clear the output vector:            
    output = np.zeros((N_out))
    # now go through all of the output neurons (indexed as after all of the hidden neurons):        
    for idx in xrange(N_hidden, N_hidden + N_out):
        # if an output neuron fired:
        if v[idx]>30:
            # reset membrane potential of fired neuron:
            v[idx] = c[idx]
            u[idx] = u[idx] + d[idx];    
   
            # record this spike as the input for the next tick:
            output[idx-N_hidden] = 1
        # end IF
        
    # integrate numerically membrane potentials for motor neurons (the last 12):
    for idx in xrange(N_hidden, N_hidden + N_out):
        # print "Accessing",idx,"-th neuron out of",N_out+N_hidden,"simulated neurons"
        for integ_tick in xrange(0, int(h)):
            # print "v[",idx,"]=",v[idx]
            # print "1/h = ",1/h
            # print "0.04*(v[",idx,"]**2)=",0.04 * (v[idx] ** 2)
            # print "5*v[",idx,"]=",5*v[idx]
            # print "140 - u[",idx,"] + I[",idx,"]=",(140 - u[idx] + I[idx])
            v[idx] += (1/h) * (0.04 * (v[idx] ** 2) + 5 * v[idx] + 140 - u[idx] + I_out[idx-N_hidden])
        # end FOR   
        #  
        # update u:
        u[idx] += a[idx] * (b[idx] * v[idx] - u[idx])
    # end FOR
    # print np.transpose(I_out)    
    return output, v, u    

# end SIMSNN
    
def getFitness(pop_member, SMloop, gate, target_rates, max_t):
    # Function evaluates SNN performance and calculates fitness value.
    # Dome conventions used here:
    # 
    # It is assumed that each population member contains weights in this preciese order:
    # Gate:
    #
    # (wts_input_hidden) are omitted due to cortex not being simulated sending firing rates from one of two "intention" neurons
    # wts_hidden_hidden
    # wts_hidden_output
    #
    # SM loop:
    # 
    # wts_gate_hidden
    # wts_inputs_hidden
    # wts_hidden_hidden
    # wts_hidden_output
    #
    # Split the pop_member and feed appropriate parts to simSNN fcn. 
    # 
    # 1. Run the network for max_t ticks and record its outputs:
    # 
    # 1.1. Simulate the gate:
    # 
    N_out = gate.num_out # number of output neurons
    N = gate.ne + gate.ni + gate.num_out  
    N_hidden = gate.ne + gate.ni       
    # init v and u using appropriate b values:
    v_gate = np.full((N, 1), -65.0)
    u_gate = np.full((N, 1), v_gate[0]*gate.b[0])
    # since b parameter is different btw neurons:
    u_gate[0:int(gate.ratio*N_hidden)] = v_gate[0] * gate.b[1]
    u_gate[int(0.8*N_hidden):N_hidden] = v_gate[0] * gate.b[2]
    #
    len_gate = gate.num_in * N_hidden + N_hidden * N_hidden + N_hidden * gate.num_out
    # to keep track of "refractoriness" of virtual cortical neurons (not simulated)
    refract = np.zeros((gate.num_in))
    # target Fq for cortical signal:
    Fq = 40.0
    # 
    # Similarly, init v and u for the SM loop:
    N_out = SMloop.num_out # number of output neurons
    N = SMloop.ne + SMloop.ni + SMloop.num_out  
    N_hidden = SMloop.ne + SMloop.ni       
    # init v and u using appropriate b values:
    v_smloop = np.full((N, 1), -65.0)
    u_smloop = np.full((N, 1), v_smloop[0]*SMloop.b[0])
    # since b parameter is different btw neurons:
    u_smloop[0:int(gate.ratio*N_hidden)] = v_smloop[0] * SMloop.b[1]
    u_smloop[int(0.8*N_hidden):N_hidden] = v_smloop[0] * SMloop.b[2]
    #
    # To keep track of motors' firings each tick:
    output_spikes_before = np.zeros((N_out)) # record spikes here before hte cue
    output_spikes_after = np.zeros((N_out)) # after the cue 
    # To keep mottors' output to feed sm loop on the next tick:
    output = np.full((N_out,1),1)


    # now go through all of the ticks:
    for tick in xrange(0, max_t):
        # ============= FIRST simulate the GATE ===============
        # get a simulated input vector from "gamma-oscillating neurons"
        inputs, refract = genSpikes(tick, Fq, refract, N)
        # 
        if tick < gate.ct: # if there is no cue yet (ct = cue time)
            gate_output, v_gate, u_gate = simSNN(pop_member[0:len_gate], gate, v_gate, u_gate, inputs)
            # if this is the first tick - jump-start the SMloop by feeding it 1s in the input
            if tick == 0:
                smloop_output = np.full((SMloop.num_in, 1), 1.0)
            # now unite gate output and SMloop's own output to feed into the SMloop:    
            output = np.concatenate(gate_output, smloop_output)
            # ============= SECOND simulate the SMLOOP ===============    
            smloop_output, v_smloop, u_smloop = simSNN(pop_member[len_gate:], SMloop, v_smloop, u_smloop, output)
        # end IF    
        # !!!! FINISH ELSE HERE and the rest. Change getBehavior accordingly !!!!   
        
        if tick < gate.ct:
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in BEFORE array"
            for i in xrange(0, N_out):
                output_spikes_before[i] += output[i]
        else:   
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in AFTER array"
            for i in xrange(0, N_out):
                output_spikes_after[i] += output[i]
        # end FOR
    # end FOR        
            

    # 2. Estimate how close were the firing rates of output neurons to the target:
    # actual firing rates:
    firing_rates_before = np.zeros((N_out))
    firing_rates_after = np.zeros((N_out))
    error_before = 0.0
    error_after = 0.0
    # go through all of the output neurons:
    for i in xrange(0, N_out):
        #
        firing_rates_before[i] = output_spikes_before[i] / (1000.0 * (max_t / 1000.0))
        # if it's an experiment with 1 or more cues:
        if SNNparam.nc>0:
            firing_rates_after[i] = output_spikes_after[i] / (1000.0 * (max_t / 1000.0))
            error_after += (target_rates.after[i] - firing_rates_after[i])**2
        # print "Firing rate[",i,"]=",firing_rates[i],"because output_spikes[",i,"]=",output_spikes[i],"/", 1000.0 * (max_t / 1000.0)
        # Calculate RMS:
        error_before += (target_rates.before[i] - firing_rates_before[i])**2

    # 4. Return fitness as 1 - RMS:
    if SNNparam.nc>0:
        fitness = 2.0 * (1.0 / (1.0 + 0.5*(np.sqrt(error_before/N_out) + np.sqrt(error_after/N_out))) - 0.5) # min(fitness) was 0.5 
    else:
        fitness = 2.0 * (1.0 / (1.0 + np.sqrt(error_before/N_out)) - 0.5) # min(fitness) was 0.5 
    # (when no output neurons fire, error is max = 1.0), therefore fitness is translated by 0.5, 
    # now min = 0.0, max = 1.0
    # in case there are several pop.members with identical fitness == 0, replace by a small random # so that parent selection algorithm acts properly (its sorting doesn't know how to deal with multiple identical entries)
    # if fitness == 0.0:
    # fitness += (2*rd.random()-1.0) / 10000.0
        # print "Fitness was 0, replaced it with", fitness
    # else:    
        #print "Fitness =", fitness, "because square root =",np.sqrt(error/N_out)
    # print "Fitness =",fitness    
    return fitness
# end getFitness    
    
def getBehavior(pop_member, SMloop, gate, target_rates, max_t):
    # Function evaluates SNN performance and calculates fitness value.

    # 1. Run the network for max_t ticks and record its outputs:
    N_out = SNNparam.num_out # number of output neurons
    output_spikes_before = np.zeros((N_out)) # record spikes here before hte cue
    output_spikes_after = np.zeros((N_out)) # after the cue
    # since no gate is simulated in this experiment, gate will be substituted by random spiking:
    N_aux = 10 # gate will have 10 output neurons (basal ganglia are converging from 10^4 to 10^2)
    # Izhikevich model settings:
    N = SNNparam.ne + SNNparam.ni + SNNparam.num_out
    N_hidden = SNNparam.ne + SNNparam.ni        
    # init excitatory neurons which will be a mix of Class 1 and Class 2:
    a = np.full((N, 1), 0.02)
    b = np.full((N, 1), -0.1)
    c = np.full((N, 1), -55.0)
    d = np.full((N, 1), 6.0)
    v = np.full((N, 1), -65.0)
    u = np.full((N, 1), v[0]*b[0])

    ratio_neur_type = 0.3 # how many Class 2 neurons are added into the mix
    for idx in xrange(0,int(ratio_neur_type*N_hidden)):
        # if rd.random() < ratio_neur_type:
        a[idx] = 0.2
        b[idx] = 0.26
        c[idx] = -65.0
        d[idx] = 0.0
        # need to update u[idx], since b[idx] changed:
        u[idx] = v[idx] * b[idx]
    # Set the last 20% of neurons to fast inhibitory params:
    for idx in xrange(int(0.8*N_hidden),N_hidden):
        a[idx] = 0.1
        b[idx] = 0.2
        c[idx] = -65.0
        d[idx] = 2.0
        # need to update u[idx], since b[idx] changed:
        u[idx] = v[idx] * b[idx]
    #end FOR            

    # now go through all of the ticks:
    for tick in xrange(0, max_t):
        # during the first tick, set all of the inputs to 1s (to jump start the SNN)
        if tick == 0:
            output = np.full((N_out,1),1)
            
        # simulate gate output:
        if tick < SNNparam.ct:
            # before the cue, generate just noise:
            # aux_input_vals = np.random.randint(2, size = (N_aux,))
            aux_input_vals = np.full((N_aux),1.0)
        elif tick > SNNparam.ct + 50.0:    
            # after 50 ms since the cue, generate just noise again:
            # aux_input_vals = np.random.randint(2, size = (N_aux,))
            aux_input_vals = np.full((N_aux),1.0)
        else:
            # right after the cue and for 50 ms after it, generate strong signal:
            aux_input_vals = np.full((N_aux), 20.0)    
        # end IF    
        output, v, u = simSNN(pop_member, SNNparam, a, b, c, d, v, u, aux_input_vals, output)
        if tick < SNNparam.ct:
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in BEFORE array"
            for i in xrange(0, N_out):
                output_spikes_before[i] += output[i]
        else:   
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in AFTER array"
            for i in xrange(0, N_out):
                output_spikes_after[i] += output[i]
        
        # print np.transpose(v[25:37])
        # print output        
        # end FOR
    # end FOR        
            

    # 2. Estimate how close were the firing rates of output neurons to the target:
    # actual firing rates:
    firing_rates_before = np.zeros((N_out))
    firing_rates_after = np.zeros((N_out))
    # go through all of the output neurons:
    for i in xrange(0, N_out):
        #
        firing_rates_before[i] = output_spikes_before[i] / (1000.0 * (max_t / 1000.0))
        if SNNparam.nc>0:
            firing_rates_after[i] = output_spikes_after[i] / (1000.0 * (max_t / 1000.0))
            print "Neuron #",i," had a firing rate BEFORE the cue",firing_rates_before[i]," (target =", target_rates.before[i]*1000.0," spikes per 1 second) and AFTER the cue", firing_rates_after[i], " (target =", target_rates.after[i]*1000.0,"spikes per 1 second)"
        else:
            print "ONLY 1(one) cue used in the training. Neuron #",i," had a firing rate =",firing_rates_before[i]," (target =", target_rates.before[i]*1000.0," spikes per 1 second)"        
        # print "Neuron's #",i," target rate was set at",target_rates[i]*1000.0," spikes per 1 second."


# end getBehavior    

def selectParents(fitness, num_par): 
    # function for selecting parents using stochastic universal sampling + identifying worst pop members to be replaced
    # check that there are no 2 elements with exactly the same fitness (leads to errors):
    # fitness_sort = np.sort(fitness)
    # print "Fitness", fitness
    # print "Sorted fitness", fitness_sort
    # print "============DEBUG: Checking for identical fitness scores==========="
    # print "Fitness =",fitness
    # print "Sorted fitness =", fitness_sort
    
    # for idx in xrange(1,len(fitness)):    
    #     if fitness_sort[idx-1]==fitness_sort[idx]:
    #         print "Found identical entries at fitness_sort[",idx,"] and fitness_sort[",idx-1, "] with the value =",fitness_sort[idx]
    #         # also change the original array:
    #         duplicate_id = np.where(fitness==fitness_sort[idx])[0]
    #         # fitness_sort[idx]=fitness_sort[idx]+0.0000001
    #         # print "Now fitness_sort[",idx,"] is", fitness_sort[idx]
    # 
    #         print "These duplicates are located at", duplicate_id, " in the original fitness array"
    #         # print "This fitness is located at", duplicate_id,"in the original fitness: fitness[",duplicate_id,"]=",fitness[duplicate_id]
    #         # now make sure that all of identical entries have slightly different fitness scores:
    #         for entry in xrange(0,len(duplicate_id)-1):
    #             print "Adding this value:",float(entry+1) / 10.0**6
    #             fitness[duplicate_id[entry]] = fitness[duplicate_id[entry]] + float(entry+1) / 10.0**6
    #             print "New fitness[",duplicate_id[entry],"] =",fitness[duplicate_id[entry]]
    #         print "Fitness[",duplicate_id[entry+1],"] remained the same:",fitness[duplicate_id[entry+1]]    
    
    for idx in xrange(0, len(fitness)):
        duplicate_ids = np.where(fitness == fitness[idx])[0]
        if len(duplicate_ids) > 1:
            for entry in xrange(0,len(duplicate_ids)-1):
                # print "Adding this value:",float(entry+1) / 10.0**6
                fitness[duplicate_ids[entry]] = fitness[duplicate_ids[entry]] + float(entry+1) / 10.0**6
                # print "New fitness[",duplicate_ids[entry],"] =",fitness[duplicate_ids[entry]]
            # print "Fitness[",duplicate_ids[entry+1],"] remained the same:",fitness[duplicate_ids[entry+1]]    
            
    # calculate normalized fitness:
    total_fitness = m.fsum(fitness)
    norm_fitness = np.zeros((len(fitness)))
    for i in xrange(0,len(fitness)):
        norm_fitness[i] = fitness[i] / total_fitness
    # end FOR
    # print "Fitness that should not contain identical entries:",fitness    
    # print "Normalized fitness:",norm_fitness
    # create cumulative sum array for stochastic universal sampling (http://www.geatbx.com/docu/algindex-02.html#P472_24607):
    cumul_fitness = [[0 for x in range(len(fitness))] for y in range(2) ] # one row for cumul fitness values, the other for population member IDs
    count = 0
    norm_fitness_temp = norm_fitness # make a copy of normalized fitness to extract indices
    while count < len(fitness):
        # find max fit and its ID:
        max_fit = max(norm_fitness)
        max_fit_id = np.argwhere(norm_fitness_temp==max_fit)

        # store cumulative norm fitness (add the sum of all previous elements)
        if count==0:
            cumul_fitness[0][count]=max_fit
            cumul_fitness[1][count]=int(max_fit_id[0])
        else:
            cumul_fitness[0][count]=max_fit+cumul_fitness[0][count-1] # have to add previous fitnesses
            cumul_fitness[1][count]=int(max_fit_id[0]) # record the ID of the pop member
        
        # remove this min fit from temp fit array:
        norm_fitness_new = np.delete(norm_fitness, norm_fitness.argmax())
        norm_fitness = norm_fitness_new
        count = count + 1
    # end WHILE
    # print "Cumul.fit[0] =",cumul_fitness[0]    
    # print "Cumul.fit[1] =",cumul_fitness[1]
    
    # roll a roulette wheel once to select 10 parents:    
    # generate the "roll" btw 0 and 1/num_parents:
    random_pointer = rd.uniform(0, 1.0/num_par)
    # keep parent IDs here:
    par_ids = []
    count = 0 # counter to go through potential parents. Keep this counter here, so that next parent is searched starting where the last one was found, not from the beginning
    par_count = 0.0 # counter for threshold adjustment
    
    # detect where roulette wheel prongs ended up:
    for x in xrange(0,num_par):
        found_flag = 0     
        while found_flag==0:
            if cumul_fitness[0][count]>(random_pointer + par_count / num_par): # for each successive parent the roulette wheel pointer shifts by 1/num_par      
                par_ids.append(cumul_fitness[1][count])
                found_flag = 1
            else:
                count = count + 1
            # end IF
        # end WHILE
        par_count = par_count + 1
    # end FOR

    # IDs of pop.members to be replaced:
    last_inx = len(cumul_fitness[1])
    # print "Cumul fitness:",cumul_fitness[0]
    # print "Cumul fitness IDs:", cumul_fitness[1]
    # print "Worst IDs start with",last_inx-num_par,"and ends with",last_inx
    worst_ids = cumul_fitness[1][last_inx-num_par:last_inx]
    return (par_ids, worst_ids) 
# end SELECTPARENTS

def produceOffspring(pop, par_ids, worst_ids, was_changed,prob_xover,prob_mut):
    # produces children using uniform crossover and bitflip mutation
    num_pairs = len(par_ids)/2
    for x in xrange(0,num_pairs):
        id1 = rd.choice(par_ids)
        del par_ids[par_ids.index(id1)] # remove the id from the list so it's not chosen twice
        id2 = rd.choice(par_ids)
        del par_ids[par_ids.index(id2)]
        # generate mask for crossover:
        mask=np.random.randint(2, size=(len(pop[id1]),))
        # generate a pair of children bitwise:
        child1=[]
        child2=[]
        if rd.uniform(0.0,1.0)<prob_xover:
            for y in xrange(0,len(pop[id1])):
                if mask[y]==0:
                    child1.append(pop[id1][y])
                    child2.append(pop[id2][y])
                else:
                    child1.append(pop[id2][y])
                    child2.append(pop[id1][y])
                # end IF
            # end FOR
        else:
            child1,child2=pop[id1],pop[id2] # if unsuccessful, children are exact copies of parents
        # end IF            
        
        # apply mutation:
        len_indv=len(pop[0])
        for y in xrange(0,len_indv):
            # roll a die for the first child:
            if rd.uniform(0.0,1.0)<prob_mut:
                if child1[y]==0: # determine the bit value and flip it
                    child1[y]=1
                else:
                    child1[y]=0
                # end IF 
            # end IF
            
            # roll a die for the second child:
            if rd.uniform(0.0,1.0)<prob_mut:
                if child2[y]==0: # determine the bit value and flip it
                    child2[y]=1
                else:
                    child2[y]=0
                # end IF
            # end IF 
        # end FOR            

        # replace individuals with lowest fitness but leave a small chance (5%) that the worst member will survive:
        if rd.uniform(0.0,1.0)<0.95:
            pop[worst_ids[x]]=child1 # Ex: for 5 pairs, replace 0-th and 5-th; 1-st and 6-th ...
            was_changed[worst_ids[x]]=0 # mark that the pop.member was changed and its fitness should be re-evaluated
        # else:
            # print "Worst_ids[",worst_ids[x],"] had survived culling and remains in the population!"
        if rd.uniform(0.0,1.0)<0.95:
            # print "Worst_ids =", worst_ids
            # print "Trying to access",x+num_pairs
            # print "Which is",worst_ids[x+num_pairs]
            # print "Pointing to",pop[worst_ids[x+num_pairs]]
            pop[worst_ids[x+num_pairs]]=child2        
            was_changed[worst_ids[x+num_pairs]]=0 # 0 - need to re-evaluate fitness 
        # else:
            # print "Worst_ids[",worst_ids[x+num_pairs],"] had survived culling and remains in the population!"    
        # end IF
    # end FOR
    return (pop,was_changed)
# end PRODUCEOFFSPRING        
                
# # # ALGORITHM # # #  
start_time = t.time()
np.random.seed(0)
# # SNN parameters:
# 
# Sensorimotor loop:
sm_num_input = 12
sm_num_output = 12
sm_num_hidden = 50
sm_num_gate = 10
sm_class12_ratio = 0.3 # ratio of Class2 to the total number of hidden neurons
N = sm_num_hidden + sm_num_output # these are actually simulated
num_cue = 0
cue_time = 1500 # this is in ms, should not be too close to the beginning or the end of the max_t
# so values in [100,900] ms for a 1000 ms simulation are better to avoid going out of bounds 
# (there is 50 ms time window when there is a high signal from the gate)
#
# gains are introduced to overcome the size limitation (to have an SNN continuously 
# drive its own activity, it needs to be several hundred neurons large):
sm_input_gain = 2.0
sm_hidden_gain = 10.0
sm_output_gain = 5.0
len_smloop = sm_num_gate * sm_num_hidden + sm_num_input * sm_num_hidden + sm_num_hidden * sm_num_hidden + sm_num_hidden * sm_num_output
SMloop = Bunch(a=[0.02, 0.2, 0.1], b=[-0.1, 0.26, 0.2], c=[-55.0, -65.0, -65.0], d=[6.0, 0.0, 2.0], ig=sm_input_gain, hg=sm_hidden_gain, og=sm_output_gain, ne=int(sm_num_hidden*0.8), ni=int(sm_num_hidden*0.2), num_in=sm_num_input, num_out=sm_num_output, length=len_smloop, nc = num_cue, ct= cue_time, ratio=sm_class12_ratio)
# a,b,c,d - Class 1, Class 2, Fast Spiking
#
# Gate:
g_num_input = 2
g_num_output = 10
g_num_hidden = 50
g_input_gain = 2.0
g_hidden_gain = 10.0
g_output_gain = 5.0
g_class12_ratio = 0.3
len_gate = g_num_input * g_num_hidden + g_num_hidden * g_num_hidden + g_num_hidden * g_num_output
# wts input (from "cortex") to hidden + wts hidden to hidden        + wts hidden to output
#
gate = Bunch(a=[0.02, 0.2, 0.1], b=[-0.1, 0.26, 0.2], c=[-55.0, -65.0, -65.0], d=[6.0, 0.0, 2.0], ig=g_input_gain, hg=g_hidden_gain, og=g_output_gain, ne=int(g_num_hidden*0.8), ni=int(g_num_hidden*0.2), num_in=g_num_input, num_out=g_num_output, length=len_gate, nc = num_cue, ct= cue_time, ratio=g_class12_ratio)
# 
# Now set target firing rates per each condition
# older version of initiation when all of the neurons produce the same rate:
#
# target_rates_val_1 = np.full((num_output, 1), 0.1) # target rates are set to be maximum for max fit
#
# newer init version when each neuron is assigned its own target firing rate:
target_rates_val_1 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
target_rates_val_2 = np.full((SMloop.num_out, 1), 0.5) # a new target to be achieved after the cue
target_rates = Bunch(before=target_rates_val_1, after=target_rates_val_2)
#
# GA parameters:
# Each individual in the evolving population will have length:
len_indv = len_smloop + len_gate
# 
# GA vars:
num_pop = 40
prob_xover = 1.0
prob_mut = 0.05
num_par = 20
max_gen = 200
max_t = 1000 # time to simulate the SNN, 1000 ms

print "Evolving vectors of length =",len_indv

# to keep track of performance:
best_fit=np.zeros((max_gen))
avg_fit=np.zeros((max_gen))
# fitness array:
fitness = np.zeros((num_pop))
# array to keep track of unchanged population members (to avoid calculating their fitness):
was_changed = np.zeros((num_pop)) # 0 - means the member was changed and needs to be re-assessed, 1 - no need to re-assess its fitness

# generate initial population:
pop = [np.random.randint(2, size=(len_indv,)) for i in xrange(0,num_pop)]

# ============ MAIN GENERATIONAL CYCLE ============ :
for gen in xrange(0,max_gen):
    print "Gen #",gen
    # Check that proper pop.members get their was_changed flags up:
    # print "Was changed =",was_changed
    print "==========================================="
    if gen > 0:
        curr_time = t.time()
        time_diff = curr_time - start_time
        # print "Time diff =",time_diff
        apprx_time_per_gen = time_diff / (gen + 1)
        # print "apprx_time_per_gen =",apprx_time_per_gen
        apprx_time_left = (max_gen - gen - 1) * apprx_time_per_gen
        # print "apprx_time_left",apprx_time_left
        hours_left = np.floor(apprx_time_left / 3600)
        mins_left = np.floor((apprx_time_left - hours_left * 3600) / 60)
        secs_left = apprx_time_left - hours_left * 3600 - mins_left * 60
        
        print hours_left, "hours", mins_left, "mins", np.round(secs_left), "seconds left..."
        print "==========================================="
        
    # calculate fitness for new population members: 
    for member in xrange(0, num_pop):
        # print "Getting fintess for #",member
        if was_changed[member]==0:
            fitness[member] = getFitness(pop[member], SMloop, gate, target_rates, max_t)               
            was_changed[member]=1
            # print "Evaluating",member,"member"
            # print "Got this fitness =",fitness[member]
        # end IF
    # end FOR
    
    # store this generation's best and avg fitness:
    best_fit[gen]=max(fitness)
    avg_fit[gen]=mean(fitness)    
    print "Best fit=",format(best_fit[gen], '.4f'),"Avg.fit=",format(avg_fit[gen], '.4f')
    print "==========================================="
    # select parents for mating:
    # print "Fitness =", fitness, ", need to choose parents #:", num_par
    (par_ids, worst_ids) = selectParents(fitness,num_par)
    print "Chose parents",par_ids,"Will replace these",worst_ids
    # for idx in xrange(0,len(worst_ids)):
        # print "Fitness[",worst_ids[idx],"]=",fitness[worst_ids[idx]]
    # replace worst with children produced by crossover and mutation:
    (pop,was_changed) = produceOffspring(pop, par_ids, worst_ids, was_changed,prob_xover,prob_mut)
    
    # inform on the performance of the best pop.member so far:
    if gen > 0:
        if np.mod(gen,10.0)==0.0:
            # best_id_temp = fitness.argsort()[-1:][::-1]
            best_id_temp = np.argmax(fitness)
            # print "Best id =",best_id_temp,". Checking: fitness[",best_id_temp,"] =",fitness[best_id_temp], "; max(fitness) =", max(fitness)
            getBehavior(pop[best_id_temp], SMloop, gate, target_rates, max_t)

    # Check that proper pop.members get their was_changed flags up:
    # print "Was changed =",was_changed
    
    # SANITY CHECK: WHY IS FITNESS DIFFERENT?
    # print "====DEBUG. Only those whose fitness was not changed are checked here ===="
    # for z in xrange(0, num_pop):
    #     if was_changed[z]==1:
    #        print "Pop. member",z,"has recorded fitness of", fitness[z], "with discrepancy of",fitness[z]-getFitness(pop[z], adj_mask, SNNparam, target_rates, max_t)

# end FOR === MAIN CYCLE ===

# # save the best guy:
# best_id = fitness.argmax()
# best_name = 'best_weights+'+folderName+".txt"
# writeWeights(best_name, pop[best_id], row_len)    

# SAVE 10 BEST:
best10_ids = fitness.argsort()[-10:][::-1]
for idx in xrange(0,len(best10_ids)):
    best_name = 'best_weights'+str(0)+str(idx+1)+".txt"
    writeWeights(best_name, pop[best10_ids[idx]], len(pop[best10_ids[idx]]))
# end FOR

# os.mkdir('Results')
# os.chdir('Results')

# FITNESS PLOT:    
#
# create a new plot:
gens=np.arange(0,max_gen)
plb.plot(gens,best_fit,'b')
plb.plot(gens,avg_fit,'r')
plb.title('Fitness plot')
plb.xlabel('Generation')
plb.ylabel('Fitness')
plb.savefig('fitness_plot.png')
plb.show()

# NEURAL DYNAMICS PLOT:
# copy DEMO to RESULTS:
# shutil.copyfile(currDir+"\\PDSTEP_demo.exe",currDir+'\\Results\\PDSTEP_demo.exe')
# shutil.copyfile(currDir+"\\PDSTEP_demo.ilk",currDir+'\\Results\\PDSTEP_demo.ilk')
# shutil.copyfile(currDir+"\\PDSTEP_demo.pdb",currDir+'\\Results\\PDSTEP_demo.pdb')
# shutil.copyfile(currDir+"\\PDSTEP_demo_x64_debug.pdb",currDir+'\\Results\\PDSTEP_demo_x64_debug.pdb')
# shutil.copyfile(currDir+"\\glut64.dll",currDir+'\\Results\\glut64.dll')
# shutil.copyfile(currDir+"\\best_weights01.txt",currDir+'\\Results\\weights.txt')

# plot spiking data and analyze
# plotSpikes(pop[0], row_len)

# END SCRIPT


