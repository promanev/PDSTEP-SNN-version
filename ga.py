import numpy as np
import math as m
import os
import sys
# VACC only: 
# import subprocess as subp
import random as rd
import time as t
import shutil, errno
import pylab as plb
# from bokeh.plotting import figure, output_file, show

# CLASSES:
# similar to C and MATLAB structures. Use as:
# Example: point = Bunch(datum=y, squared = y*y, coord x)
# point.isok = 1, etc.
#
class Bunch:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)
# end CLASS BUNCH        

# FUNCTIONS:
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

def simSNN(pop_member, SNNparam, a, b, c, d, v, u, aux_input_vals, output):
    # Function that creates an SNN and runs it for 1 sim. step
    # aux_inputs - spikes that came from another subnetwork
    # num_output - number of special neurons that communicate outside of the current subnetwork
    # 
    # 
    # Begin: 
    input_gain = 30.0
    aux_gain = 30.0
    Ne = SNNparam.ne # number of excitatory hidden neurons
    Ni = SNNparam.ni # number of inhibitory hidden neurons
    N_hidden = Ne + Ni # total number of hidden neurons
    N_out = SNNparam.num_out # number of output neurons
    N_in = SNNparam.num_in # number of input neurons
    N_aux = len(aux_input_vals) # number of additional, auxilary to this SNN, inputs
    
    # convert pop_member into weight matrices:
    # gate_hidden_mark = N_aux * N_hidden
    # sensor_hidden_mark = gate_hidden_mark + N_in*N_hidden
    # hidden_hidden_mark = sensor_hidden_mark + N_hidden*N_hidden
    # hidden_motor_mark = hidden_hidden_mark + N_out * N_hidden
    
    # Wgh = pop2wts(pop_member[0:gate_hidden_mark], N_aux)
    # Wsh = pop2wts(pop_member[gate_hidden_mark:sensor_hidden_mark], N_in)
    # Whh = pop2wts(pop_member[sensor_hidden_mark:hidden_hidden_mark], N_hidden)
    # Whm = pop2wts(pop_member[hidden_hidden_mark:hidden_motor_mark], N_out)
    #
    # in order to reduce complexity, I will only optimize hid-hid weights
    hidden_hidden_mark = N_hidden*N_hidden
    # 
    Whh = pop2wts(pop_member[0:hidden_hidden_mark], N_hidden)
    # Whm = pop2wts(pop_member[hidden_hidden_mark:hidden_motor_mark], N_out)
    # set the rest of the matrices to 1s:
    Wgh = np.ones((N_hidden, N_aux))
    Wsh = np.ones((N_hidden, N_in))
    Whm = np.ones((N_out, N_hidden))
    #
    h = 2.0
    # create influence vector by feeding it sensor and gate data:
    I = np.zeros((N_hidden))
    # create influence vector for output neurons:
    I_out = np.zeros((N_out))
    # since weights and spikes are binary if either weight or spike are zeros - one may skip to the next input
    for i in xrange(0, N_hidden):
        # auxilary:
        for j in xrange(0, N_aux):
            if Wgh[i][j]>0:
                if aux_input_vals[j]>0:
                    I[i] += 1 * aux_gain
        # sensory:
        for j in xrange(0, N_in):
            if Wsh[i][j]>0:
                if output[j]>0:
                    I[i] += 1 * input_gain           
    # end FOR
   
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
            for idx2 in xrange(0, N_hidden):
                if Whh[idx][idx2]>0:
                    if idx > int(0.8*N_hidden):
                        I[idx2] = I[idx2] - 1.0
                    else:
                        I[idx2] = I[idx2] + 1.0
                    # end IF
                # end IF  
            # end FOR
                        
            # similarly, adjust I_out:
            for idx2 in xrange(0, N_out):
                if Whm[idx2][idx]>0:
                    if idx > int(0.8*N_hidden):
                        I_out[idx2] = I_out[idx2] - 1.0
                    else:
                        I_out[idx2] = I_out[idx2] + 1.0
                    # end IF
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
        
    # integrate numerically membrane potentials:
    for idx in xrange(N_hidden, N_hidden + N_out):
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
        
    return output, v, u    

# end SIMSNN
    
def getFitness(pop_member, SNNparam, target_rates, max_t):
    # Function evaluates SNN performance and calculates fitness value.

    # 1. Run the network for max_t ticks and record its outputs:
    N_out = SNNparam.num_out # number of output neurons
    output_spikes = np.zeros((N_out))
    # since no gate is simulated in this experiment, gate will be substituted by random spiking:
    N_aux = 10 # gate will have 10 output neurons (basal ganglia are converging from 10^4 to 10^2)
    aux_input_vals = np.random.randint(2, size = (N_aux,))
    # Izhikevich model settings:
    N = SNNparam.ne + SNNparam.ni + SNNparam.num_out        
    # init excitatory neurons which will be a mix of Class 1 and Class 2:
    a = np.full((N, 1), 0.02)
    b = np.full((N, 1), -0.1)
    c = np.full((N, 1), -55.0)
    d = np.full((N, 1), 6.0)
    v = np.full((N, 1), -65.0)
    u = np.full((N, 1), v[0]*b[0])

    ratio_neur_type = 0.3 # how many Class 2 neurons are added into the mix
    for idx in xrange(0,int(0.8*N)):
        if rd.random() < ratio_neur_type:
            a[idx] = 0.2
            b[idx] = 0.26
            c[idx] = -65.0
            d[idx] = 0.0
            # need to update u[idx], since b[idx] changed:
            u[idx] = v[idx] * b[idx]
    # Set the last 20% of neurons to fast inhibitory params:
    for idx in xrange(int(0.8*N),N):
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
        # end IF    
        output, v, u = simSNN(pop_member, SNNparam, a, b, c, d, v, u, aux_input_vals, output)
        for i in xrange(0, N_out):
            output_spikes[i] += output[i]
        # end FOR
    # end FOR        
            

    # 2. Estimate how close were the firing rates of output neurons to the target:
    # actual firing rates:
    firing_rates = np.zeros((N_out))
    error = 0.0
    # go through all of the output neurons:
    for i in xrange(0, N_out):
        #
        firing_rates[i] = output_spikes[i] / (1000.0 * (max_t / 1000.0))
        # print "Firing rate[",i,"]=",firing_rates[i],"because output_spikes[",i,"]=",output_spikes[i],"/", 1000.0 * (max_t / 1000.0)
        # Calculate RMS:
        error += (target_rates[i] - firing_rates[i])**2
    # 4. Return fitness as 1 - RMS:
    fitness = 1.0 / (1.0 + np.sqrt(error/N_out)) - 0.5
    if fitness == 0.0:
        fitness = rd.random() / 10000.0
        # print "Fitness was 0, replaced it with", fitness
    # else:    
        #print "Fitness =", fitness, "because square root =",np.sqrt(error/N_out)
    return fitness
# end getFitness    

def selectParents(fitness, num_par): 
    # function for selecting parents using stochastic universal sampling + identifying worst pop members to be replaced
    # calculate normalized fitness:
    total_fitness = m.fsum(fitness)
    norm_fitness = np.zeros((len(fitness)))
    for i in xrange(0,len(fitness)):
        norm_fitness[i] = fitness[i] / total_fitness
    # end FOR

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
        if rd.uniform(0.0,1.0)<0.95:
            # print "Worst_ids =", worst_ids
            # print "Trying to access",x+num_pairs
            # print "Which is",worst_ids[x+num_pairs]
            # print "Pointing to",pop[worst_ids[x+num_pairs]]
            pop[worst_ids[x+num_pairs]]=child2        
            was_changed[worst_ids[x+num_pairs]]=0 # 0 - need to re-evaluate fitness 
        # end IF
    # end FOR
    return (pop,was_changed)
# end PRODUCEOFFSPRING

def mean(numbers): # Arithmetic mean fcn
    return float(sum(numbers)) / max(len(numbers), 1)
# end MEAN fcn

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
        
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
# end COPYANYTHING        
                
# # # ALGORITHM # # #  
start_time = t.time()
np.random.seed(0)
# # SNN parameters:
# Sensorimotor loop:
num_input = 12
num_output = 12
num_hidden = 20
num_gate = 10
N = num_hidden + num_output # these are actually simulated



# GA parameters:
# len_indv = num_gate * num_hidden + num_input * num_hidden + num_hidden * num_hidden + num_hidden * num_output
# # -----    wts btw gate & hid  + wts btw input & hidden + wts btw hidden & hidden  +  wts btw hidden & output
len_indv = num_hidden * num_hidden
num_pop = 50
prob_xover = 1.0
prob_mut = 0.05
num_par = 10
max_gen = 50
max_t = 1000 # time to simulate the SNN, 1000 ms

# to keep track of performance:
best_fit=np.zeros((max_gen))
avg_fit=np.zeros((max_gen))
# fitness array:
fitness = np.zeros((num_pop))
# array to keep track of unchanged population members (to avoid calculating their fitness):
was_changed = np.zeros((num_pop)) # 0 - means the member was changed and needs to be re-assessed, 1 - no need to re-assess its fitness

# generate initial population:
pop = [np.random.randint(2, size=(len_indv,)) for i in xrange(0,num_pop)]

# generate SNN parameters
# parameter structure per sub-network, it's only one here, since the gate is not yet added
SNNparam = Bunch(ne=16, ni=4, num_in=12, num_out=12 )
# end FOR   
target_rates = np.full((SNNparam.num_out, 1), 0.1) # target rates are set to be maximum for max fit   

# main cycle:
for gen in xrange(0,max_gen):
    print "Gen #",gen
               
    # calculate fitness for new population members: 
    for member in xrange(0, num_pop):
        if was_changed[member]==0:
            fitness[member] = getFitness(pop[member], SNNparam, target_rates, max_t)               
            was_changed[member]=1
 #           print "Evaluating",member,"member"
        # end IF
    # end FOR
    
    # store this generation's best and avg fitness:
    best_fit[gen]=max(fitness)
    avg_fit[gen]=mean(fitness)    
    print "Best fit=",best_fit[gen],"Avg.fit=",avg_fit[gen]
    # select parents for mating:
    # print "Fitness =", fitness, ", need to choose parents #:", num_par
    (par_ids, worst_ids) = selectParents(fitness,num_par)
 #   print "Chose parents",par_ids,"Will replace these",worst_ids
    #for idx in xrange(0,len(worst_ids)):
    #    print "Fitness[",worst_ids[idx],"]=",fitness[worst_ids[idx]]
    # replace worst with children produced by crossover and mutation:
    (pop,was_changed) = produceOffspring(pop, par_ids, worst_ids, was_changed,prob_xover,prob_mut)
    
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

os.mkdir('Results')
os.chdir('Results')

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
