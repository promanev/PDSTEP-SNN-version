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
import os
import sys
import subprocess as subp

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
    
def readTextFile(fileName):
    loadFlag = False
    loadAttempt = 0
    attemptMax = 10
    dataVar = []
    while (loadAttempt<attemptMax) & (loadFlag==False):
        if(os.path.exists(fileName)==False):
            # pause for 0.5 second:
            t.sleep(0.5)
            loadAttempt = loadAttempt + 1
            print("Load attempt "+str(loadAttempt))
        else:
            fid = open(fileName,'r')
            # read just one line from fit.txt
            i=0
            for line in fid:
                dataVar += [float(line)]
            # end FOR
            fid.close()
            loadFlag = True
        # end IF

        # if can't find the fit.txt file for a long time:
        if loadAttempt >= attemptMax:
            sys.exit('Can''t load '+fileName)
        # end IF
    # end WHILE
    # end IF
    return dataVar
# end DEF

def writeTextFile(fileName,data):
    fid = open(fileName,'w+')
# check if data is 1 variable or list/array:
    if (isinstance(data,float) or isinstance(data,int)):
        fid.write(str(data))
    else:
        for j in xrange(0,len(data)):
            fid.write(str(data[j])+"\n")
        # end FOR
    # end IF
    fid.close()
# end DEF

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
        
def genSpikes(tick, Fq, refract):
    # Fcn generates one-dim vector of inputs size from a simulated rhythm  
    # Since there is a source of stochasticity here, it's replaced for allowing
    # a fair comparison btw pop members. 
    #
    # number of neurons to be simulated:
    N = len(refract)
    # vector to output:
    outputs = np.zeros((N))      
    # period = 1000/Fq
    # refractory period in ms:
    # refract_value = int(period*0.9)
    # refract_value = period
    # the closer you are to this threshold the higher the probability of a spike:
    # threshold = float(float(tick%period)/period)
    # 
    # print "Tick",tick,"threshold =",threshold
    for n in xrange(0,N):
        #
        # if rd.random() < threshold:
        # if threshold == 0:
        #     # record spike and make neuron refractory            
        #    if refract[n]==0.0:
        #        refract[n]=refract_value
        #        outputs[n]=1
        #    # end IF
        # # end IF
        # # 
        # # print refract        
        # # check if there are refractory periods, and decrease them:
        # if refract[n]>0.0:
        #     refract[n]=refract[n]-1.0
        # # end IF
        outputs[n]=1.0
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
        # print "Simulating SM loop..."
    else:
        is_smloop = 0
        # print "Simulating Gate..."
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
        # print "Wgh = [",len(Wgh),",",len(Wgh[0]),"]"
        # print "Wsh = [",len(Wsh),",",len(Wsh[0]),"]"
        # print "Whh = [",len(Whh),",",len(Whh[0]),"]"
        # also break down inputs from gate and sm loop itself in two:
        aux_input_vals = inputs[0:N_aux]
        smloop_input = inputs[N_aux:]
        # print "Gate->hidden length", gate_hidden_mark
        # print "Inputs->hidden length", sensor_hidden_mark
        # print "Hidden->hidden length", hidden_hidden_mark
    else:
        sensor_hidden_mark = N_in*N_hidden
        hidden_hidden_mark = sensor_hidden_mark + N_hidden*N_hidden
        Wsh = pop2wts(pop_member[0:sensor_hidden_mark], N_in)
        Whh = pop2wts(pop_member[sensor_hidden_mark:hidden_hidden_mark], N_hidden)
        # print "Wsh = [",len(Wsh),",",len(Wsh[0]),"]"
        # print "Whh = [",len(Whh),",",len(Whh[0]),"]"
        # print "Inputs->hidden length", sensor_hidden_mark
        # print "Hidden->hidden length", hidden_hidden_mark
        
    # end IF    
    
    hidden_motor_mark = hidden_hidden_mark + N_out * N_hidden    
    Whm = pop2wts(pop_member[hidden_hidden_mark:hidden_motor_mark], N_out)
    # print "Whm = [",len(Whm),",",len(Whm[0]),"]"
    # print "Hidden->motors length",hidden_motor_mark

    # Izhikevich model's parameters:
    # Start with setting all of neurons to Class 1 Excitatory 
    # (the last N_out neurons - output neurons - will stay this kind):
    a = np.full((N, 1), SNNparam.a[0])
    b = np.full((N, 1), SNNparam.b[0])
    c = np.full((N, 1), SNNparam.c[0])
    d = np.full((N, 1), SNNparam.d[0])
    # v = np.full((N, 1), -65.0)
    # u = np.full((N, 1), v[0]*b[0])
    # print "Setting principal neurons: a=",a[0],"b=",b[0],"c=",c[0],"d=",d[0]
    
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
    # print "Setting secondary principal neurons: a=",a[0],"b=",b[0],"c=",c[0],"d=",d[0]
    # print "Setting inhibitory neurons: a=",a[int(0.8*N_hidden+1)],"b=",b[int(0.8*N_hidden+1)],"c=",c[int(0.8*N_hidden+1)],"d=",d[int(0.8*N_hidden+1)]    
    # print "Checking that output neurons are the same as principal neurons: a=",a[-1],"b=",b[-1],"c=",c[-1],"d=",d[-1]
    #
    h = 2.0
    # create influence vector by feeding it sensor and gate data:
    I = np.zeros((N_hidden))
    # create influence vector for output neurons:
    I_out = np.zeros((N_out))
    #
    if is_smloop: # SMLOOP SIMULATED
        # print "SM loop. i ->", N_hidden,"j (aux) ->",N_aux,"j (in) ->",N_in
        for i in xrange(0, N_hidden):
            # auxilary:
            for j in xrange(0, N_aux):
                if Wgh[i][j]>0:
                    if aux_input_vals[j]>0:
                        I[i] += 1 * Wgh[i][j]
            # sensory:
            for j in xrange(0, N_in):
                if Wsh[i][j]>0:
                    if smloop_input[j]>0:
                        I[i] += 1 * Wsh[i][j]           
        # end FOR
    else: # GATE SIMULATED!!!
        # print "Gate. i ->", N_hidden,"j (in) ->",N_in     
        for i in xrange(0, N_hidden):               
           for j in xrange(0, N_in):
               # print "Inputs[",j,"]=",inputs[j]
               if Wsh[i][j]>0:
                   if inputs[j]>0:
                       if SNNparam.cue_flag: # cue_flag is raised if a cue was introduced and for several ticks afterwards, this is calculated in getFitness()
                           if i < int(0.8 * (SNNparam.ni + SNNparam.ne)): # if this is an excitatory gate neuron
                               I[i] += 1 * Wsh[i][j]
                           else: # if this is an INHIBITORY gate neuron (gets increased drive after the cue to cause beta ERD)
                               I[i] += 1 * Wsh[i][j] * 5
                           # end IF    
                       else: # if we are outside of "motor change" (i.e., before or sufficiently later after the cue)
                           I[i] += 1 * Wsh[i][j]
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
                        I[idx2] = I[idx2] - 1.0 * Whh[idx][idx2]
                    else:
                        I[idx2] = I[idx2] + 1.0 * Whh[idx][idx2]
                    # end IF
                    # print "AFTER I[", afferent_id,"]=",I[afferent_id]   
                # end IF  
            # end FOR
                        
            # similarly, adjust I_out:
            for idx2 in xrange(0, N_out):
                if Whm[idx][idx2]>0:
                    if idx > int(0.8*N_hidden):
                        I_out[idx2] = I_out[idx2] - 1.0 * Whm[idx][idx2]
                    else:
                        I_out[idx2] = I_out[idx2] + 1.0 * Whm[idx][idx2]
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
    
def getFitness(pop_member, SMloop, gate, target_rates, max_t, behavior_flag):
    # Multi-objective fitness function evaluates SNN performance and calculates two fitness value.
    # The first value is based on how the actual firings match the targets.
    # The second value is based on how great is the difference between firing modes before and after the cue
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
    N_ctx_neurons = gate.num_in       
    # init v and u using appropriate b values:
    v_gate = np.full((N, 1), -65.0)
    u_gate = np.full((N, 1), v_gate[0]*gate.b[0])
    # since b parameter is different btw neurons:
    u_gate[0:int(gate.ratio*N_hidden)] = v_gate[0] * gate.b[1]
    u_gate[int(0.8*N_hidden):N_hidden] = v_gate[0] * gate.b[2]
    #
    len_gate = gate.num_in * N_hidden + N_hidden * N_hidden + N_hidden * gate.num_out
    # to keep track of "refractoriness" of virtual cortical neurons (not simulated)
    refract = np.zeros((int(N_ctx_neurons / 2))) # only one cortical neuron is simulated per condition
    # print refract,"cortical neurons emulated per active action representation (neurons for other action are set to 0)"
    # target Fq for cortical signal:
    Fq_1 = 100.0 # cortical signal for action 1
    Fq_2 = 200.0 # cortical signal for action 2
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
    # To keep motors' output to feed sm loop on the next tick:
    output = np.full((N_out,1),1)
    # 
    #
    # read the target firing rates and determine baseline fitness score an SNN would
    # get if it did not fire at all:
    # base_fit = 0
    # if gate.nc > 0:
    #     base_fit_after = 0
    # # end IF    
    # for i in xrange(0, N_out):
    #     base_fit += target_rates.before[i]**2     
    #     if gate.nc > 0:
    #         base_fit_after += target_rates.after[i]**2
    #     # end
    # base_fit = 1 / (1 + np.sqrt(base_fit / N_out))
    # print "Based on target rates, the base fitness is", base_fit
    # if gate.nc > 0:
    #     base_fit_after = 1 / (1 + np.sqrt(base_fit_after / N_out))
    # # end IF    

    # now go through all of the ticks:
    for tick in xrange(0, max_t):
        # ============= FIRST simulate the GATE ===============
        # 
        if (tick >= gate.ct) and (tick < gate.ct + 100): # if the cue just happened, then gate should have an increased drive to its inh neurons     
            # which has been shown before to increase gamma band of the spectrum 
            # This cue_flag makes input to inh neurons of the gate increase 5 fold        
            gate.cue_flag = 1
            # get a simulated input vector from "gamma-oscillating neurons"
            ctx_spikes, refract = genSpikes(tick, Fq_2, refract)
            # print "!!!Beta ERD!!! Tick =",tick,"(cue time =",gate.ct,"); Fq2 =", Fq_2, "; refract =", refract
            # print "Cortical neuron output =", ctx_spikes
        else: # if there is no cue yet (ct = cue time) or its at least 100 ms after the cue
            gate.cue_flag = 0
            # get a simulated input vector from "gamma-oscillating neurons"
            ctx_spikes, refract = genSpikes(tick, Fq_1, refract)
            # print "!NO Beta ERD! Tick =",tick,"(cue time =",gate.ct,"); Fq1 =", Fq_1, "; refract =", refract
            # print "Cortical neuron output =", ctx_spikes
        # end IF
                    
        # form the output from "cortex" consisting of two neurons (one per action):
        if tick < gate.ct:
            inputs=np.concatenate((ctx_spikes[0:len(refract)], np.zeros((len(refract)))))
            # print "BEFORE the cue. Cortex should have 1st neuron group active only:", inputs
        else:            
            inputs=np.concatenate((np.zeros((len(refract))), ctx_spikes[0:len(refract)]))
            # print "AFTER the cue. Cortex should have 2nd neuron group active only:", inputs
        #
        # Now, run the gate for 1 tick:    
        gate_output, v_gate, u_gate = simSNN(pop_member[0:len_gate], gate, v_gate, u_gate, inputs)
        # 
        # now move to simulate the SM loop.
        # if this is the first tick - jump-start the SMloop by feeding it 1s in the input
        if tick == 0:
            smloop_output = np.full((SMloop.num_in), 1.0)
        # end IF
        #    
        # now unite gate output and SMloop's own output to feed into the SMloop: 
        output = np.concatenate((gate_output, smloop_output))
        
        # ============= SECOND simulate the SMLOOP ===============    
        smloop_output, v_smloop, u_smloop = simSNN(pop_member[len_gate:], SMloop, v_smloop, u_smloop, output)
        # end IF  
        
        # print "Gate output", gate_output
        # print "SMloop output", smloop_output
        # print "v_smloop =", v_gate
        # print "v_smloop =", v_smloop
        
        if tick < gate.ct:
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in BEFORE array"
            for i in xrange(0, N_out):
                output_spikes_before[i] += smloop_output[i]
        else:   
            # print "Tick",tick,"is below the cue time (",SNNparam.ct,"ms), recording in AFTER array"
            for i in xrange(0, N_out):
                output_spikes_after[i] += smloop_output[i]
        # end IF
    # end FOR        

    # 2. Estimate how close were the firing rates of output neurons to the target:
    # actual firing rates:
    firing_rates_before = np.zeros((N_out))
    fit_before = 0 # this value will be multiplied by individual fitnesses of each neuron, so having 1s as initail value is ok
    if gate.nc > 0:
        fit_after = 0
        firing_rates_after = np.zeros((N_out))
        fitness_2 = 0
    # end IF

    # go through all of the output neurons:
    for i in xrange(0, N_out):
        #
        firing_rates_before[i] = output_spikes_before[i] / (1000.0 * (max_t / 1000.0))
        #        
        # Calculate error per each neuron:
        temp_val_before = 0.0
        if target_rates.before[i] > firing_rates_before[i]:
            if firing_rates_before[i] == 0:           
                # print "Encountered 0.0 firing rate, adding 1 spike (0.001 rate)..."
                temp_val_before = target_rates.before[i] / (firing_rates_before[i] + 0.001)
            else:
                temp_val_before = target_rates.before[i] / firing_rates_before[i]
            # print "T > A, temp_val_before =",temp_val_before
        else: 
            temp_val_before = firing_rates_before[i] / target_rates.before[i]           
            # print "A > T, temp_val_before =",temp_val_before
        # fit_before = fit_before + ( 1.0 / (1.0 + np.sqrt( (target_rates.before[i] - firing_rates_before[i])**2 ) / target_rates.before[i] ) - 0.5 )
        fit_before = fit_before + ( 1.0 / (1.0 + temp_val_before) )

        # if it's an experiment with 1 or more cues:
        if gate.nc>0:
            firing_rates_after[i] = output_spikes_after[i] / (1000.0 * (max_t / 1000.0))
            temp_val = 0.0
            if target_rates.after[i] > firing_rates_after[i]:
                if firing_rates_after[i]==0:
                    temp_val = target_rates.after[i] / (firing_rates_after[i] + 0.001)
                else:                
                    temp_val = target_rates.after[i] / firing_rates_after[i]
                # end IF
            else:
                temp_val = firing_rates_after[i] / target_rates.after[i]
            # end IF                
            fit_after = fit_after + ( 1.0 / (1.0 + temp_val) )
            #
            # get the second fitness value (based on the difference of firing before and after):
            fitness_2 += np.sqrt( (firing_rates_before[i] - firing_rates_after[i])**2 )

            if behavior_flag:
                print "Neuron ",i," had a firing rate BEFORE the cue",firing_rates_before[i]*1000," (target =", target_rates.before[i]*1000.0," spikes per 1 second) and AFTER the cue", firing_rates_after[i], " (target =", target_rates.after[i]*1000.0,"spikes per 1 second)"
                print "Fitness 1 (matching target firing rates) before the cue =",1.0/(1.0 + temp_val_before)
                print "Fitness 1 (matching target firing rates) after the cue =",1.0/(1.0 + temp_val)
                print "Fitness 2 (maximize difference btw firing rates before and after the cue) =", np.sqrt( (firing_rates_before[i] - firing_rates_after[i])**2 )
            # end IF
        else:
            if behavior_flag:
                print "ONLY 1(one) cue used in the training. Neuron #",i," had a firing rate =",firing_rates_before[i]*1000," (target =", target_rates.before[i]*1000.0," spikes per 1 second)"        

        # DEBUG
        # print "Output neuron",i,":"
        # print "Actual fir.rate =",firing_rates_before[i],"/",target_rates.before[i],"(its indv. target)"
        # print "temp_val_before =", temp_val_before
        # print "1 / (1 + temp_val)", 1.0 / (1.0 + temp_val_before)

    # 4. Return fitness as (1/(1 + RMS) - 0.5) * 2. Shorter version 1/(1+RMS) 
    # was augmented because it has a min of 0.5 and max 1.0, so I am translating 
    # fitness score by 0.5 and scale up the results so now fitness should be
    # within [0,1]:
    # print "Current population member acquired error score of", error_before
    if gate.nc>0:
        fitness = min(fit_before,fit_after) / N_out # if there is a cue, the lowest of two fitnesses is considered the results (more selective pressure: those that excel at both tasks get a higher fitness, not those that specialize only in one)
        fitness_2 = fitness_2 / N_out
    else:
        fitness = fit_before / N_out
        # print "Its fitness before subtraction is:", 1.0 / (1.0 + np.sqrt(error_before/N_out))
        # print "After baselining the fitness:", fitness
    # if fitness == 0.0:
    # fitness += (2*rd.random()-1.0) / 10000.0
        # print "Fitness was 0, replaced it with", fitness
    # else:    
        #print "Fitness =", fitness, "because square root =",np.sqrt(error/N_out)
    # print "Fitness =",fitness    
    return (fitness, fitness_2)
# end getFitness    
    
def selectParents(fitness, num_par): 
    # function for selecting parents using stochastic universal sampling + identifying worst pop members to be replaced
    # check that there are no 2 elements with exactly the same fitness (leads to errors):
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

def bloat_pop(pop, fitness, fitness_2, prob_xover, prob_mut, SMloop, gate, target_rates, max_t):
    # This fcn is increasing the population twofold and applies mutation operator to the new guys. 
    # Then it also evaluates them and updates fitness and fitness_2 arrays.
    # 
    # get the size of population:
    pop_size = len(pop)
    # print "Population size is",pop_size
    indv_size = len(pop[0])
    # print "Each indv is of size",indv_size
    # get the max value that pop. vector can take:
    pop_max_val = np.max(pop)+1 # +1 because otherwise random values will be generated in [0, N-1]
    # print "Max val of synaptic weight",pop_max_val
    # check that N is even, if not reduce by 1 (this is only to make sure that there are enough pairs):
    if np.mod(pop_size,2)!=0:
        pop_size = pop_size - 1
    # end IF
    #
    # make two arrays to keep fitness scores for offspring:
    fit_offspring = np.zeros(pop_size)
    fit2_offspring = np.zeros(pop_size)
    #
    # create a random ordering list that will be used to choose parents:
    parent_order = np.random.permutation(np.arange(pop_size))
    #
    # now go through this list appending new population members to pop array and simultaneously calculating their fitnesses:
    for p_parent in xrange(0,int(pop_size/2)):
        # generate mask for crossover:
        mask=np.random.randint(2, size=indv_size)
        # generate a pair of children bitwise:
        child1=[]
        child2=[]
        id1 = parent_order[p_parent]
        id2= parent_order[p_parent+1]
        # if a succesful cast of dice:
        if rd.uniform(0.0,1.0)<prob_xover:
            for y in xrange(0,indv_size):
                if mask[y]==0:
                    child1.append(pop[id1][y])
                    child2.append(pop[id2][y])
                else:
                    child1.append(pop[id2][y])
                    child2.append(pop[id1][y])
                # end IF
            # end FOR Y
        else:
            child1,child2=pop[id1],pop[id2] # if unsuccessful, children are exact copies of parents
        # end IF            
        
        # apply mutation:
        for y in xrange(0,indv_size):
            # roll a die for the first child:
            if rd.uniform(0.0,1.0)<prob_mut:
                mut_value = np.random.randint(pop_max_val,size=1)
                child1[y]=mut_value[0]
            # end IF
            
            # roll a die for the second child:
            if rd.uniform(0.0,1.0)<prob_mut:
                mut_value = np.random.randint(pop_max_val,size=1)
                child2[y]=mut_value[0]
            # end IF 
        # end FOR Y    
        # 
        # when the first pair of children created, init a temp.array to keep children in:
        if p_parent == 0:
            offspring = child1
            offspring = np.concatenate((offspring,child2))
#            offspring = np.reshape(offspring, (2, indv_size))
        else: # and continue concatenating children from subsequent parents:
            offspring = np.concatenate((offspring,child1))
            offspring = np.concatenate((offspring,child2))
        # end IF P_PARENT
        #
        # now evaluate children on two fitness objectives:
        # print "Child1=",child1
        (fit_offspring[p_parent], fit2_offspring[p_parent]) = getFitness(child1, SMloop, gate, target_rates, max_t, 0)
        # print "Child2=",child2
        (fit_offspring[p_parent+1], fit2_offspring[p_parent+1]) = getFitness(child2, SMloop, gate, target_rates, max_t, 0)
    # end FOR P_PARENT
    #
    # now append offspring to the major population and their respective fitness scores to fitness vectors:
    #print "Pop arrays is",len(pop),"x",len(pop[0])
    #print "Offspring array is",len(offspring)
    offspring = np.reshape(offspring, (pop_size, indv_size))
    #print "Offspring array is resized to",len(offspring),"x",len(offspring[0])
    new_pop = np.concatenate((pop, offspring))
    new_fitness = np.concatenate((fitness,fit_offspring))              
    new_fitness2 = np.concatenate((fitness_2,fit2_offspring))              
    # output new arrays:
    return (new_pop, new_fitness, new_fitness2)
# end BLOAT_POP

def cull_the_weak(pop, fitness, fitness_2):
    # This fcn is conducting pairwise tournaments to weed out those population members that perform equally bad on both objectives
    # until the original population size is reached (half of population that is passed into this function)
    #
    # get the size of population vector being evolved:
    N = len(pop)
    # print "Pop.size is",N
    # check that N is even, if not reduce by 1 (this is only to make sure that there are enough pairs):
    if np.mod(N,2)!=0:
        N = N - 1
        print "Pop. size was odd, reduced it by 1 to",N,"to ensure that there will be a pair for everyone"
    # end IF
    #
    # create a random ordering list that will be used to choose competitors:
    pop_ids = np.arange(N)
    competitors_order = np.random.permutation(pop_ids)
    # print "Order of competition:",competitors_order
    # create a variable to keep track of how many pairs of competitors are left:
    competitor_pairs_left = int(N/2)
    # ... and which pairs are competing currently:
    p_compet =0
    # create a variable to keep track of current population size:
    curr_pop_size = N  
    # instead of deleting those who lost, include them in a special list:
    delete_these = []
    # protection from when there is no possible winner:
    num_attempts = N * 20
    # conduct tournaments until the target population size is reached:
    while (curr_pop_size > int(N/2)) & (num_attempts > 0):
        print "Tournaments left:",num_attempts
        num_attempts -= 1
        # pick competitors:
        id1 = competitors_order[p_compet]
        id2 = competitors_order[p_compet+1]
        # print "Competing",id1,"vs.",id2
        compet_1 = pop[id1]
        compet_2 = pop[id2]
        # compare fitness scores:
        compare_1 = fitness[id1] > fitness[id2]
        compare_2 = fitness_2[id1] > fitness_2[id2]
        # print "Fitness comparison:",fitness[id1],"vs.",fitness[id2]
        # print "Fitness_2 comparison:",fitness_2[id1],"vs.",fitness_2[id2]
        # First check that both fitness scores are not equal:
        if ( fitness[id1] <> fitness[id2] ) & ( fitness_2[id1] <> fitness_2[id2] ):
            # now decide who stays and who dies:
            if compare_1: # if TRUE this means that id1 has greater fitness than id2
                if compare_2: # if TRUE - id1 has greater fitness_2 than id2
                    # since id1 is better id2 on both objectives - delete id2:
                    # pop = np.delete(pop, (id2), axis=0)
                    # fitness = np.delete(fitness, (id2), axis=0)
                    # fitness_2 = np.delete(fitness_2, (id2), axis=0)
                    curr_pop_size -= 1
                    delete_these.append(id2)
                    where_pop_id = np.where(pop_ids==id2)
                    pop_ids = np.delete(pop_ids, (where_pop_id[0]), axis=0)
                    # print "Member",id2,"is worse on both objectives, removing..."
                    # print "Updated pop. size", curr_pop_size
                    del where_pop_id
                # end IF COMPARE_2 - no else because if each competitor excels at a different objective - both stay
            elif compare_2 != True:
                # now the opposite is true - id2 is better, kill id1:
                # pop = np.delete(pop, (id1), axis=0)
                # fitness = np.delete(fitness, (id1), axis=0)
                # fitness_2 = np.delete(fitness_2, (id1), axis=0)
                delete_these.append(id1)
                where_pop_id = np.where(pop_ids==id1)
                pop_ids = np.delete(pop_ids, (where_pop_id[0]), axis=0)
                curr_pop_size -= 1
                # print "Member",id1,"is worse on both objectives, removing..."
                # print "Updated pop. size", curr_pop_size
                del where_pop_id
            # end IF COMPARE_2
        elif fitness[id1] == fitness[id2]:
            if fitness_2[id1] > fitness_2[id2]:
                # Both excel equally good on 1st objective, but id1 is better on the second, kill id2
                # pop = np.delete(pop, (id2), axis=0)
                # fitness = np.delete(fitness, (id2), axis=0)
                # fitness_2 = np.delete(fitness_2, (id2), axis=0)
                delete_these.append(id2)
                where_pop_id = np.where(pop_ids==id2)
                pop_ids = np.delete(pop_ids, (where_pop_id[0]), axis=0)
                curr_pop_size -= 1
                # print "Member",id2,"is worse on FITNESS_2 only, removing..."
                # print "Updated pop. size", curr_pop_size
                del where_pop_id
            elif fitness_2[id1] < fitness_2[id2]: # the opposite outcome - kill id1
                # pop = np.delete(pop, (id1), axis=0)
                # fitness = np.delete(fitness, (id1), axis=0)
                # fitness_2 = np.delete(fitness_2, (id1), axis=0)
                delete_these.append(id1)
                where_pop_id = np.where(pop_ids==id1)
                pop_ids = np.delete(pop_ids, (where_pop_id[0]), axis=0)
                curr_pop_size -= 1
                # print "Member",id1,"is worse on FITNESS_2 only, removing..."
                # print "Updated pop. size", curr_pop_size
                del where_pop_id
            # end IF FITNESS_2...
        # end IF <>
 
        # shift pointer by 2 so it'll point to the 1st guy in the next pair:
        p_compet += 2
        # reduce the number of pairs left in the current permutation:
        competitor_pairs_left -= 1
        # check if there is at least one pair left, if not - generate new permutation:
        if competitor_pairs_left < 1:
            # now generate permutation using indices of those not yet deleted stored in dynamic var. pop_ids:
            competitors_order = np.random.permutation(pop_ids)
            p_compet = 0
            # check if current permutation vector is even, if not - discard the first member from new permutation 
            # (?might lead to one very unfit individual staying? but since every time this is encountered it will be a different guy the bias should be gone):
            if np.mod(len(competitors_order),2)!=0:
                competitors_order = np.delete(competitors_order, (0), axis=0)
            # end IF NP.MOD
            # 
            competitor_pairs_left = len(competitors_order) / 2
        # end IF COMPETITOR_PAIRS
        # 
        # DEBUG:
        # print "Curr pop size",curr_pop_size
        # print "Left pairs in this permutation",competitor_pairs_left        
    # end WHILE
    #
    print "Chose these to delete",delete_these
    # if WHILE was interrupted by attempts running out:
    if num_attempts < 1:
        if curr_pop_size > N/2:
            # then remove enough children (likely to be worse than the main population) to make it to target size:
            more_to_remove = curr_pop_size - N/2
            print "Tournaments ran out but still need to weed out",more_to_remove,"pop.members"
            delete_these.extend(pop_ids[-more_to_remove:])
            print "Appended list of those to be deleted:",delete_these
    # now delete those that lost tournament:
    # print "BEFORE deletion, there are",len(pop)," members,",len(fitness),"fit scores,",len(fitness_2),"fit_2 scores..."
#    for idx in delete_these:
    pop = np.delete(pop, (delete_these), axis=0)
    fitness = np.delete(fitness, (delete_these), axis=0)
    fitness_2 = np.delete(fitness_2, (delete_these), axis=0)
    # print "AFTER deletion, there are",len(pop)," members,",len(fitness),"fit scores,",len(fitness_2),"fit_2 scores..."
        
    return pop, fitness, fitness_2

# end CULL_THE_WEAK
                
# # # ALGORITHM # # #  
# 
# VACC only:
os.chdir("/gpfs1/home/r/p/rpopov/EXP8_Disemb_model/Full_model_one_targ_MO/")
EXP_ID = "EXP8_FULL_1T_MO"
timeStamp = t.localtime()
uniq = int(rd.random()*100)
folderName = EXP_ID+"-"+str(timeStamp.tm_mday)+"-"+str(timeStamp.tm_mon)\
            +"-"+str(timeStamp.tm_year)+"-"+ str(timeStamp.tm_hour)\
            +"-"+str(timeStamp.tm_min)+"-"+str(timeStamp.tm_sec)+"-"+str(uniq)

os.mkdir(folderName)
# read seeds from seeds.txt:
seedVal = readTextFile('seeds.txt')
# set RNG to use the seed
rd.seed(int(seedVal[0]))
print("Using seed: "+str(int(seedVal[0])))
# store the rest of seeds aside from value used (1st in the array):
writeTextFile('seeds.txt', seedVal[1:])
# go to experiment directory:
os.chdir(folderName)

# store this run's seed:
writeTextFile('this_seed.txt', seedVal[0])

start_time = t.time()
# # SNN parameters:
# 
# Sensorimotor loop:
sm_num_input = 12
sm_num_output = 12
sm_num_hidden = 20
sm_num_gate = 10
sm_class12_ratio = 0.3 # ratio of Class2 to the total number of hidden neurons
N = sm_num_hidden + sm_num_output # these are actually simulated
num_cue = 1
cue_time = 500 # this is in ms, should not be too close to the beginning or the end of the max_t
# so values in [100,900] ms for a 1000 ms simulation are better to avoid going out of bounds 
# (there is 50 ms time window when there is a high signal from the gate)
#
# gains are introduced to overcome the size limitation (to have an SNN continuously 
# drive its own activity, it needs to be several hundred neurons large):
sm_input_gain = 2.0
sm_hidden_gain = 5.0
sm_output_gain = 5.0
len_smloop = sm_num_gate * sm_num_hidden + sm_num_input * sm_num_hidden + sm_num_hidden * sm_num_hidden + sm_num_hidden * sm_num_output
SMloop = Bunch(a=[0.02, 0.2, 0.1], b=[-0.1, 0.26, 0.2], c=[-55.0, -65.0, -65.0], d=[6.0, 0.0, 2.0], ig=sm_input_gain, hg=sm_hidden_gain, og=sm_output_gain, ne=int(sm_num_hidden*0.8), ni=int(sm_num_hidden*0.2), num_in=sm_num_input, num_out=sm_num_output, length=len_smloop, nc = num_cue, ct= cue_time, ratio=sm_class12_ratio)
# a,b,c,d - Class 1, Class 2, Fast Spiking
#
# Gate:
#
# number of neurons:
g_num_input = 20
g_num_output = 10
g_num_hidden = 10
# gains
g_input_gain = 2.0
g_hidden_gain = 5.0
g_output_gain = 5.0
g_class12_ratio = 0.3
len_gate = g_num_input * g_num_hidden + g_num_hidden * g_num_hidden + g_num_hidden * g_num_output
# wts input (from "cortex") to hidden + wts hidden to hidden        + wts hidden to output
#
print "SM loop len =",len_smloop, ". Gate len =", len_gate
gate = Bunch(a=[0.02, 0.2, 0.1], b=[-0.1, 0.26, 0.2], c=[-55.0, -65.0, -65.0], d=[6.0, 0.0, 2.0], ig=g_input_gain, hg=g_hidden_gain, og=g_output_gain, ne=int(g_num_hidden*0.8), ni=int(g_num_hidden*0.2), num_in=g_num_input, num_out=g_num_output, length=len_gate, nc = num_cue, ct= cue_time, ratio=g_class12_ratio, cue_flag=0)
# 
# Now set target firing rates per each condition
# older version of initiation when all of the neurons produce the same rate:
#
# target_rates_val_1 = np.full((num_output, 1), 0.1) # target rates are set to be maximum for max fit
#
# newer init version when each neuron is assigned its own target firing rate:
target_rates_val_1 = [0.05, 0.1, 0.12, 0.14, 0.15, 0.16, 0.175, 0.185, 0.195, 0.205, 0.215, 0.225]
target_rates_val_2 = [0.1, 0.2, 0.24, 0.28, 0.3, 0.32, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45] # a new target to be achieved after the cue
target_rates = Bunch(before=target_rates_val_1, after=target_rates_val_2)
#
# GA parameters:
# Each individual in the evolving population will have length:
len_indv = len_smloop + len_gate
# 
# GA vars:
num_pop = 30
prob_xover = 1.0
prob_mut = 0.05
num_par = 2
max_gen = 500
max_t = 1000 # time to simulate the SNN, 1000 ms

print "Evolving vectors of length =",len_indv

# to keep track of performance:
best_fit=np.zeros(max_gen)
avg_fit=np.zeros(max_gen)
best_fit2 = np.zeros(max_gen)
avg_fit2 = np.zeros(max_gen)
# fitness array:
fitness = np.zeros(num_pop)
fitness_2 = np.zeros(num_pop)
# array to keep track of unchanged population members (to avoid calculating their fitness):
# was_changed = np.zeros((num_pop)) # 0 - means the member was changed and needs to be re-assessed, 1 - no need to re-assess its fitness

# generate initial population:
pop = [np.random.randint(50, size=(len_indv,)) for i in xrange(0,num_pop)]
# calculate fitness for the population members: 
for member in xrange(0, num_pop):
    # print "Getting fintess for #",member
    # if was_changed[member]==0:
    (fitness[member], fitness_2[member]) = getFitness(pop[member], SMloop, gate, target_rates, max_t, 0)               
    # was_changed[member]=1
    # print "Evaluating",member,"member"
    # print "Got this fitness =",fitness[member]
# end FOR MEMBER

# ============ MAIN GENERATIONAL CYCLE ============ :
for gen in xrange(0,max_gen):
    print "Gen",gen+1,"/",max_gen
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
    # end IF GEN    
    # 
    # Create new offspring effectively doubling the population:
    (new_pop, new_fitness, new_fitness_2) = bloat_pop(pop, fitness, fitness_2, prob_xover, prob_mut, SMloop, gate, target_rates, max_t)
    # Using tournament selection kull the weak until the population is back to original size:
    (pop, fitness, fitness_2) = cull_the_weak(new_pop, new_fitness, new_fitness_2)
    # store this generation's best and avg fitness:
    best_fit[gen]=max(fitness)
    avg_fit[gen]=mean(fitness)    
    best_fit2[gen]=max(fitness_2)
    avg_fit2[gen]=mean(fitness_2)    
    print "Best fit=",format(best_fit[gen], '.8f'),"Avg.fit=",format(avg_fit[gen], '.8f')
    print "Best fit2=",format(best_fit2[gen], '.8f'),"Avg.fit2=",format(avg_fit2[gen], '.8f')
    print "==========================================="
    # select parents for mating:
    # print "Fitness =", fitness, ", need to choose parents #:", num_par
    # (par_ids, worst_ids) = selectParents(fitness,num_par)
    # print "Chose parents",par_ids,"Will replace these",worst_ids
    # for idx in xrange(0,len(worst_ids)):
        # print "Fitness[",worst_ids[idx],"]=",fitness[worst_ids[idx]]
    # replace worst with children produced by crossover and mutation:
    # (pop,was_changed) = produceOffspring(pop, par_ids, worst_ids, was_changed,prob_xover,prob_mut)
    
    # inform on the performance of the best pop.member so far:
    if gen > 0:
        if np.mod(gen,10.0)==0.0:
            # best_id_temp = fitness.argsort()[-1:][::-1]
            best_id_temp_1 = np.argmax(fitness)
            best_id_temp_2 = np.argmax(fitness_2)
            print "Best in matching firing rates:"
            # print "Best id =",best_id_temp,". Checking: fitness[",best_id_temp,"] =",fitness[best_id_temp], "; max(fitness) =", max(fitness)
            getFitness(pop[best_id_temp_1], SMloop, gate, target_rates, max_t, 1)               
            print "Best in making diverse firing rates:"
            getFitness(pop[best_id_temp_2], SMloop, gate, target_rates, max_t, 1)               
            # save current best solutions:
            writeWeights('best_wts_fit1_'+str(gen)+'.txt', pop[best_id_temp_1],len(pop[0]))
            writeWeights('best_wts_fit2_'+str(gen)+'.txt', pop[best_id_temp_2],len(pop[0]))

    # Check that proper pop.members get their was_changed flags up:
    # print "Was changed =",was_changed
    
    # SANITY CHECK: WHY IS FITNESS DIFFERENT?
    # print "====DEBUG. Only those whose fitness was not changed are checked here ===="
    # for z in xrange(0, num_pop):
    #     if was_changed[z]==0:
    #      # print "Pop. member",z,"has recorded fitness of", fitness[z], "with discrepancy of",fitness[z]-getFitness(pop[z], adj_mask, SNNparam, target_rates, max_t)
    #      print "Pop #",z,":",pop[z][1:10]  
# end FOR === MAIN CYCLE ===

# SAVE 10 BEST ON FIT1:
best10_ids = fitness.argsort()[-10:][::-1]
for idx in xrange(0,len(best10_ids)):
    best_name = 'best_weights_fit1_'+str(0)+str(idx+1)+".txt"
    writeWeights(best_name, pop[best10_ids[idx]], len(pop[best10_ids[idx]]))
writeWeights('best_fit.txt',best_fit,len(best_fit))
writeWeights('avg_fit.txt',avg_fit,len(avg_fit))
# end FOR
# SAVE 10 BEST ON FIT2:
best10_ids = fitness_2.argsort()[-10:][::-1]
for idx in xrange(0,len(best10_ids)):
    best_name = 'best_weights_fit2_'+str(0)+str(idx+1)+".txt"
    writeWeights(best_name, pop[best10_ids[idx]], len(pop[best10_ids[idx]]))
writeWeights('best_fit2.txt',best_fit2,len(best_fit))
writeWeights('avg_fit2.txt',avg_fit2,len(avg_fit))
# end FOR

# os.mkdir('Results')
# os.chdir('Results')

# FITNESS PLOT:    
#
# create a new plot:
# gens=np.arange(0,max_gen)
# plb.plot(gens,best_fit,'b')
# plb.plot(gens,avg_fit,'r')
# plb.title('Fitness plot')
# plb.xlabel('Generation')
# plb.ylabel('Fitness')
# plb.savefig('fitness_plot.png')
# plb.show()

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


