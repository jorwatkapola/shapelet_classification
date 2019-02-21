from __future__ import division
import importlib
import shapelets as sha
import os
import fnmatch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import csv
from sklearn import tree
import sys
sys.stdout.flush()
import math

#set the data directory path depending on the workstation being used
cwd = os.getcwd()
if cwd.split("/")[1] == "home":
    data_path="/home/jkok1g14/Documents/GRS1915+105/data/Std1_PCU2"
elif cwd.split("/")[1] == "export":
    data_path="/export/data/jakubok/GRS1915+105/Std1_PCU2"
else:
    print("Set the path for data directory!", Flush=True)

#load the file with data labels (observation ID against the class of activity)
clean_belloni = open('1915Belloniclass_updated.dat')
lines = clean_belloni.readlines()
states = lines[0].split()
belloni_clean = {}
for h,l in zip(states, lines[1:]):
    belloni_clean[h] = l.split()
    #state: obsID1, obsID2...
ob_state = {}
for state, obs in belloni_clean.items():
    #merge the chi classes of different variability into one
    if state == "chi1" or state == "chi2" or state == "chi3" or state == "chi4": state = "chi"
    for ob in obs:
        ob_state[ob] = state
        #keys of observation IDs, single item with the classification

available = []
pool=[]

#find the available data for the classified observations
for root, dirnames, filenames in os.walk(data_path):
    for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
        available.append(filename)
for ob, state in ob_state.items():
    if ob+"_std1_lc.txt" in available:
        pool.append(ob)  

#create a list of arrays with time and counts for the set of Belloni classified observations
lc_dirs=[]
lcs=[]
ids=[]
for root, dirnames, filenames in os.walk(data_path):    
    for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
        if filename.split("_")[0] in pool:
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
    #f8c=f8c-np.mean(f8c)#normalisation/mean centering/whatever you desire most
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
            

####### due to the energy integration in Std1 diefferences between different epochs shouldn't matter; there would be very few photons found at the extremes of the range            
    times=np.delete(f8t,rm_points)
    counts=np.delete(f8c,rm_points)
    lcs.append(np.stack((times,counts)))
#a list of light curve 2D arrays

lc_classes=[]
for i in ids:
    lc_classes.append(ob_state[i])

drop_classes=[]
for clas, no in Counter(lc_classes).items():
    if no<7:
        drop_classes.append(clas)

lcs_abu = []
classes_abu = []
ids_abu = []
for n, lc in enumerate(lc_classes):
    if lc not in drop_classes:
        classes_abu.append(lc)
        lcs_abu.append(lcs[n])
        ids_abu.append(ids[n])  
#a list of light curve 2D arrays of classes with at least 7 light curves

lcs_abu_std=sha.scaling(lcs_abu, method="standard")
# data is standardised, x_i_stand = (x_i - x_mean)/x_std
# mean+n*sigma is going to be the assumed maximum count rate that will be used to normalise the data
        
x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(lcs_abu, classes_abu, ids_abu, test_size=0.5, random_state=0, stratify=classes_abu)

best_shapelets=[]
time_res=1
for n_donor, lc_donor in enumerate(x_train):
    #Create lists with classifications of all time-series relative to the donor time series; one that the pool of shapelets is generated from
    state_donor = y_train[n_donor]
    belong_class=[]
    other_class=[]
    for n, i in enumerate(id_train):
        if y_train[n] == state_donor:
            belong_class.append(i)
        else:
            other_class.append(i)
    #calculate the entropy of the entire set, so it can be compared to the split set later
    prop_belong = len(belong_class)/(len(belong_class)+len(other_class))
    set_entropy = -(prop_belong)*math.log(prop_belong, 2)-(1-prop_belong)*math.log(1-prop_belong, 2)
    pool=sha.generate_shapelets(lc_donor, 1, len(lc_donor[0]))#generate shapelets from the donor time-series, 
    #set the initial best value of information gain to 0 (improved by any split) and start testing the shapelets
    best_gain=0
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
            if id_train[n_donor] == n_lc:
                distance = 0
            else:
                lc=x_train[np.where(np.array(id_train)==n_lc)[0][0]]
                distance=sha.distance_calculation(shapelet, lc, early_abandon=False)
            #save the distance value together with the classification and lightcurve id
            if n_lc in belong_class:
                class_assign=1
            else:
                class_assign=0
            distances.append((n_lc ,distance, class_assign))
            #find the optimal split point if there are at least two distances calculated, then use entropy pruning to find if the shapelet still has a change to beat the best one found so far
            if len(distances)>1:
                best_split=sha.best_split_point(distances, set_entropy)
                skip_shapelet=sha.entropy_pruning(best_gain, distances, best_split, len(belong_class), len(other_class), set_entropy)
                if skip_shapelet==True:
                    break
        #if shapelet was not rejected at entropy pruning, calculcate the information gain and if the value is larger then the best one so far, save the shapelet
        if skip_shapelet==False:
            gain=sha.information_gain(distances, set_entropy, best_split)
            if gain>best_gain:
                best_gain=gain
                best_shapelet=shapelet
    print((best_shapelet, best_split, best_gain, id_train[n_donor], state_donor))
    best_shapelets.append((best_shapelet, best_split, best_gain, id_train[n_donor], state_donor))

train_dists=np.zeros((len(x_train),len(best_shapelets)))
for i_t, t in enumerate(x_train):
    for i_s, s in enumerate(best_shapelets):
        distance=sha.distance_calculation(s[0], t, early_abandon=False)
        train_dists[i_t,i_s]=distance
print("train_dists\n", train_dists)

test_dists=np.zeros((len(x_test),len(best_shapelets)))
for i_t, t in enumerate(x_test):
    for i_s, s in enumerate(best_shapelets):
        distance=sha.distance_calculation(s[0], t, early_abandon=False)
        test_dists[i_t,i_s]=distance
print("test_dists\n",test_dists)


dtc=tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(train_dists, y_train)
train_inference= dtc.score(train_dists)
print("train_inference_score\n",train_inference)

dtc.fit(train_dists, y_train)
inference= dtc.predict(test_dists)
print("test_inference\n", inference)
score=[]
for n, i in enumerate(inference):
    if i ==y_test[n]:
        score.append(1)
    else:
        score.append(0)
score=np.array(score)
print("test_inference_score\n",np.mean(score))

