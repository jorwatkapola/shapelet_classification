import sys
sys.stdout.flush()
import os
import fnmatch
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import math
import shapelets as sha
from operator import itemgetter

#create a belloni_files list with names of files holding Belloni classified light curves
clean_belloni = open(file_name)
lines = clean_belloni.readlines()
states = lines[0].split()
belloni_clean = {}
for h,l in zip(states, lines[1:]):
    belloni_clean[h] = l.split()
    #state: obsID1, obsID2...
ob_state = {}
for state, obs in belloni_clean.items():
    if state == "chi1" or state == "chi2" or state == "chi3" or state == "chi4": state = "chi"
    for ob in obs:
        ob_state[ob] = state
return ob_state
#xp10408013100_lc.txt classified as chi1 and chi4, xp20402011900_lc.txt as chi2 and chi2
#del ob_state["10408-01-31-00{}".format(extension)] as long as training and test sets are checked for duplicates when appending, it should be ok to keep

#create lists of available light curves and those with available lables
available = []
pool=[]
extension="_std1_lc.txt"
for root, dirnames, filenames in os.walk("/home/jkok1g14/Documents/GRS1915+105/data/Std1_PCU2"):
    for filename in fnmatch.filter(filenames, "*{}".format(extension)):
        available.append(filename)
for ob, state in ob_state.items():
    if ob in available:
        pool.append(ob)        

#split the observations into training and test sets
training_obs = []
training_states = []
test_obs = []
test_states = []
randomize = np.random.choice(list(ob_state.keys()), len(ob_state.keys()), replace=False)
for ob in randomize:
    state = ob_state["{}".format(ob)]
    if state not in training_states:
        if ob in pool:
            training_obs.append(ob)
            training_states.append(state)
no_train = math.ceil(len(pool)*0.50)
for ob in training_obs:
    pool.remove("{}".format(ob))        
for ob in randomize:
    state = ob_state["{}".format(ob)]
    if state not in test_states:
        if ob in pool:
            test_obs.append(ob)
            test_states.append(state)
for ob in test_obs:
    pool.remove("{}".format(ob))
remaining = int(no_train-len(training_obs))
train_remain = np.random.choice(pool, size = remaining, replace=False)
for ob in train_remain:
    training_obs.append(ob)
for ob in pool:
    if ob not in training_obs:
        test_obs.append(ob)
        
        
#create a list of arrays with time and counts for the set of Belloni classified observations
lc_dirs=[]
lcs=[]
ids=[]
#for root, dirnames, filenames in os.walk("/export/data/jakubok/GRS1915+105/Std1_PCU2"):
for root, dirnames, filenames in os.walk("/home/jkok1g14/Documents/GRS1915+105/data/Std1_PCU2"):    
    for filename in fnmatch.filter(filenames, "*{}".format(extension)):
        if filename.split("_")[0] in list(ob_state.keys()):
            lc_dirs.append(os.path.join(root, filename))

            
#make 2D arrays for light curves, with columns of counts and time values
for lc in lc_dirs:
    ids.append(lc.split("/")[-1].split("_")[0])
    f=np.loadtxt(lc)
    f=np.transpose(f)#,axis=1)
    f=f[0:2]
    ###1s average and time check to eliminate points outside of GTIs
    f8t = np.mean(f[0][:(len(f[0])//8)*8].reshape(-1, 8), axis=1)
    f8c = np.mean(f[1][:(len(f[1])//8)*8].reshape(-1, 8), axis=1)
    rm_points = []
    skip=False
    for i in range(len(f8t)-1):
        if skip==True:
            skip=False
            continue
        delta = f8t[i+1]-f8t[i]
        if delta > 1.0:
            rm_points.append(i+1)
            skip=True
            
####### normalise the count rates! think about the effect of 0-1 normalisation on the distance calculation
            
    times=np.delete(f8t,rm_points)
    counts=np.delete(f8c,rm_points)
    lcs.append(np.stack((times,counts)))
    break
print("No. of light curves prepared: {}".format(len(lcs)))













#create a pool of shapelets for the set of light curves, best_shapelets contains one shapelet from every light curve that produced the greatest information gain
tested_classes=[]
best_shapelets=[]
time_res=1
for n_donor, lc_donor in enumerate(lcs):
    #Create lists with classifications of all time-series relative to the donor time series; one that the pool of shapelets is generated from
    state_donor = ob_state[ids[n_donor]]
    if state_donor not in tested_classes:
        tested_classes.append(state_donor)
    else:
        continue
    belong_class=[]
    other_class=[]
    for n_lc in range(len(lcs)):
        if ob_state[ids[n_lc]] == state_donor:
            belong_class.append(n_lc)
        else:
            other_class.append(n_lc)
    #calculate the entropy of the entire set, so it can be compared to the split set later
    prop_belong = len(belong_class)/(len(belong_class)+len(other_class))
    set_entropy = -(prop_belong)*math.log2(prop_belong)-(1-prop_belong)*math.log2(1-prop_belong)
    pool=sha.generate_shapelets(lc_donor, 1, len(lc_donor[0]))#generate shapelets from the donor time-series, 
    best_gain=0#set the initial best value of information gain to 0 (improved by any split) 
    for shapelet in pool:
        skip_shapelet=False#for entropy pruning
        #set the order of distance calculations
        #pick an other_class object first and then alternate between belong and other, when one group runs out, append the rest of the other group to the end
        order=[]
        if len(belong_class)<len(other_class):alternations=len(belong_class);larger_group=other_class
        else: alternations=len(other_class); larger_group=belong_class
        for i in range(alternations):
            order.append(other_class[i])
            order.append(belong_class[i])
        for i in range(len(larger_group)-alternations):
            order.append(larger_group[-(i+1)])
        #start distance calculations
        distances=[]
        for n_lc in order:
            lc=lcs[n_lc]
            distance=sha.distance_calculation(n_lc, lc, shapelet, time_res, belong_class)
           # print(n_donor, distance, shapelet)
            distances.append(distance)
            if len(distances)>1:
                best_split=sha.best_split_point(distances, set_entropy)
                skip_shapelet=sha.entropy_pruning(best_gain, distances, best_split, len(belong_class), len(other_class), set_entropy)
                if skip_shapelet==True:
                    break
        if skip_shapelet==False:
            gain=sha.information_gain(distances, set_entropy, best_split)
            #print(shapelet)
            #print(distances)
            if gain>best_gain:
                best_gain=gain
                best_shapelet=shapelet
    best_shapelets.append((best_shapelet, best_gain, state_donor))