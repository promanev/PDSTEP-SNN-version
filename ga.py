import numpy as np
import math as m
import os
import sys
# import subprocess as subp
import random as rd
import time as t
# from shutil import copy2 as cp
# import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show

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

def getFitness(pop_member, wtsFileName, row_len):
#    writeWeights(wtsFileName, pop_member, row_len)                
#    subp.call(['./RagdollDemoApp'])
#    # read fit:
#    dataRead = readTextFile('fit.txt')
#    fitness = dataRead[0]
#    os.remove('fit.txt')
    fitness=m.fsum(pop_member)/len(pop_member)+rd.random()*0.01
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
   # print "Total fitness=",total_fitness
   # print "Normalized fitness=",norm_fitness
    # create cumulative sum array for stochastic universal sampling (http://www.geatbx.com/docu/algindex-02.html#P472_24607):
    cumul_fitness = [[0 for x in range(len(fitness))] for y in range(2) ] # one row for cumul fitness values, the other for population member IDs
    count = 0
    norm_fitness_temp = norm_fitness # make a copy of normalized fitness to extract indices
    while count < len(fitness):
        # find max fit and its ID:
        max_fit = max(norm_fitness)
        max_fit_id = np.argwhere(norm_fitness_temp==max_fit)
    #    print "Count=",count,"Max_fit=",max_fit,"located at",max_fit_id[0]
    #    print "Double-check:",norm_fitness_temp[max_fit_id[0]],"Length of temp array=",len(norm_fitness_temp)
        # store cumulative norm fitness (add the sum of all previous elements)
        if count==0:
            cumul_fitness[0][count]=max_fit
            cumul_fitness[1][count]=int(max_fit_id[0])
        else:
            cumul_fitness[0][count]=max_fit+cumul_fitness[0][count-1] # have to add previous fitnesses
            cumul_fitness[1][count]=int(max_fit_id[0]) # record the ID of the pop member
        # remove this min fit from temp fit array:
     #   print "Norm_fit before deletion",norm_fitness
        norm_fitness_new = np.delete(norm_fitness, norm_fitness.argmax())
        norm_fitness = norm_fitness_new
     #   print "Norm_fit after deletion",norm_fitness
        count = count + 1
    # end WHILE
   # print "Cumul_fitness:",cumul_fitness[0]
   # print "Indices:",cumul_fitness[1]
        
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
    worst_ids = cumul_fitness[1][last_inx-num_par:last_inx]
    #print "Should be selected to be worst",worst_ids
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
            pop[worst_ids[x+num_pairs]]=child2        
            was_changed[worst_ids[x+num_pairs]]=0 # 0 - need to re-evaluate fitness 
        # end IF
    # end FOR
    return (pop,was_changed)
# end PRODUCEOFFSPRING

def mean(numbers): # Arithmetic mean fcn
    return float(sum(numbers)) / max(len(numbers), 1)
# end MEAN fcn    
                
# # # ALGORITHM # # #   
# output to static HTML file
output_file("log_lines.html")                
  
# get current directory path:
currDir = os.getcwd()

# create a new, unique folder for the run:
EXP_ID = "TEST"
timeStamp = t.localtime()
# Experiment ID tag has timestamp a random number (to make each folder have unique name)
uniq = int(rd.random()*1000)
folderName = EXP_ID+"-"+str(timeStamp.tm_mday)+"-"+str(timeStamp.tm_mon)\
            +"-"+str(timeStamp.tm_year)+"-"+ str(timeStamp.tm_hour)\
            +"-"+str(timeStamp.tm_min)+"-"+str(timeStamp.tm_sec)+"-"+str(uniq)
# create unique experiment directory:
# VACC os.mkdir(folderName)

# VACC # copy simulation file there:
# VACC cp("./RagdollDemoApp", folderName+"/RagdollDemoApp") # VACC only

# VACC # read seeds from seeds.txt:
# VACC seedVal = readTextFile('seeds.txt')
# VACC # set RNG to use the seed
# VACC rd.seed(int(seedVal[0]))
# VACC print("Using seed: "+str(int(seedVal[0])))
# VACC # store the rest of seeds aside from value used (1st in the array):
# VACC writeTextFile('seeds.txt', seedVal[1:])
# VACC # go to experiment directory:
# VACC os.chdir(folderName)

# VACC # store this run's seed:
# VACC writeTextFile('this_seed.txt', seedVal[0])

# # SNN parameters:
# num_sens = 2
# num_neur = 20
# row_len = num_neur + num_sens + 1 # the length of a weight matrix row
row_len = 20

# GA parameters:
# len_indv = num_neur * (num_neur + num_sens + 1) # +1 for a sign bit: 1 = +, 0 = -
len_indv = 20
num_pop = 200
prob_xover = 0.1
prob_mut = 0.05
num_par = 40
max_gen = 500
fileName = "fit.txt"
wtsFileName = "weights.txt"
execName = "./RagdollDemoApp"

# to keep track of performance:
best_fit=np.zeros((max_gen))
avg_fit=np.zeros((max_gen))
# fitness array:
fitness = np.zeros((num_pop))
# array to keep track of unchanged population members (to avoid calculating their fitness):
was_changed = np.zeros((num_pop)) # 0 - means the member was changed and needs to be re-assessed, 1 - no need to re-assess its fitness

# generate initial population:
pop = [np.random.randint(2, size=(len_indv,)) for i in xrange(0,num_pop)]

# main cycle:
for gen in xrange(0,max_gen):
    print "Gen #",gen
    # calculate fitness for new population members:
 #   print "Was changed:",was_changed
    for member in xrange(0, num_pop):
        if was_changed[member]==0:
            fitness[member]=getFitness(pop[member],wtsFileName,row_len)
            was_changed[member]=1
 #           print "Evaluating",member,"member"
        # end IF
    # end FOR
 #   print "Fitness=",fitness
    # store this generation's best and avg fitness:
    best_fit[gen]=max(fitness)
    avg_fit[gen]=mean(fitness)    
    print "Best fit=",best_fit[gen],"Avg.fit=",avg_fit[gen]
    # select parents for mating:
    (par_ids, worst_ids) = selectParents(fitness,num_par)
 #   print "Chose parents",par_ids,"Will replace these",worst_ids
    #for idx in xrange(0,len(worst_ids)):
    #    print "Fitness[",worst_ids[idx],"]=",fitness[worst_ids[idx]]
    # replace worst with children produced by crossover and mutation:
    (pop,was_changed) = produceOffspring(pop, par_ids, worst_ids, was_changed,prob_xover,prob_mut)
    
# end FOR
    
# PLOTS:    
#
# create a new plot:
# p = figure(
#   tools="pan,box_zoom,reset,save",
#   title="Best and Avg Fitness",
#   x_axis_label='Generations', y_axis_label='Fitness Score'
#)
# gens=np.arange(0,max_gen)
# p.circle(gens, best_fit, legend="y=x", fill_color="white", size=8)

# END SCRIPT
