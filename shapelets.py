import numpy as np
import math
from operator import itemgetter
from copy import deepcopy

def generate_shapelets(light_curve, minlen, maxlen, time_res=1.):
    """Create a list of all possible subsegments from light_curve, which is a 2D array, where [0,:] are the time values in seconds, and [1,:] are the count rate values. minlen and maxlen are the lower and upper limits of subsegment length in the unit of seconds. Output is a list of 1D arrays.The shortest shapelet will be an array of length minlen+1 etc. time_res is the time interval between two data points in seconds, so that for time_res=0.5 and minlen=100, the smallest produced shapelet would be an array of length (100/0.5)+1.
    """
    pool=[]
    lc = light_curve
    minlen=int(minlen/time_res); maxlen=int(maxlen/time_res)
    for l in range(minlen,maxlen+1):
        end=l+1; start=0
        while end<=len(lc[0]):
            sh=lc[1,start:end]#end is exclusive
            inter=(lc[0,end-1]-lc[0,start])#end is inclusive, hence the -1
            end+=1; start+=1
            if inter/time_res==float(l):#time difference between the first and last data points (inter in seconds) divided by the time difference between two consecutive data points (time_res in seconds) must be equal to the expected length of the moving window (l in seconds)
                pool.append(sh)
    return pool

def information_gain(distances, set_entropy, split_point):
    """Calculate the information gain from splitting the dichotomous set of time-series objects into subsets below and above the split_point, depending on their distances to the tested shapelet. "distances" is a list of tuples, where each tuple corresponds to a time-series object, and distances[i][0] is the id number of the ith object, distances[i][1] is the minimal distance between the shapelet and the object, distances[i][2] is equal to 1 if the object belongs to the tested object class, or to 0 otherwise (other_class objects). Information gain is dependent on the entropy of the entire set and the split set, where entropy is I(set)=-Alog2(A)-Blog2(B) (A=proportion of object in the set that belong to the class, B=proportion of other objects). Information gain is then I(set)-
    """
    try:
        above=[lc for lc in distances if lc[1]>=split_point]
        above_belong=sum([lc[2] for lc in above])
        below=[lc for lc in distances if lc[1]<split_point]
        below_belong=sum([lc[2] for lc in below])
        prop_above_belong=above_belong/len(above)
        prop_below_belong=below_belong/len(below)
        #entropy of the subgroup above the split point
        if prop_above_belong==1. or prop_above_belong==0.:
            above_entropy=0
        else:
            above_entropy = -(prop_above_belong)*math.log(prop_above_belong, 2)-(1-prop_above_belong)*math.log(1-prop_above_belong, 2)
        #entropy of the subgroup below the split point
        if prop_below_belong==1. or prop_below_belong==0.:
            below_entropy =0
        else:
            below_entropy = -(prop_below_belong)*math.log(prop_below_belong, 2)-(1-prop_below_belong)*math.log(1-prop_below_belong, 2)
        #return the information gain as the difference between the entropy of the entire set and the sum of entropies of subgroups generated from that set
        return set_entropy-(len(above)/(len(distances)))*(above_entropy)-(len(below)/(len(distances)))*(below_entropy)
    except ZeroDivisionError:
        return "Invalid split point."

def distance_calculation(shapelet, lc, time_res=1., early_abandon=False):
    """finds minimal distance between a light curve and a shapelet. lc is the light curve 2d array (lc[0] time values and lc[1] count rate values), time_res can be changed if the time resolution of the time-series is different than 1s (needs to be checked to make sure that distance is calculated only within good time intervals),  
    """
    if early_abandon==False:
        best_dist=np.inf
        lc_l = len(lc[0])
        sha_l=len(shapelet)
        for start_p in range(lc_l-sha_l+1):
            end_p=start_p+sha_l-1
            if lc[0,end_p]-lc[0,start_p] != (sha_l-1)*time_res:
                continue
            sha_dist=np.sum((shapelet-lc[1,start_p:end_p+1])**2)
            #sha_dist=0
            #for i in range(sha_l):
                #sha_dist += (lc[1,i+start_p]-shapelet[i])**2
            if sha_dist<best_dist:
                best_dist=sha_dist
        return (best_dist)
    else:
        best_dist=np.inf 
        lc_l = len(lc[0])
        sha_l=len(shapelet)
        for start_p in range(lc_l-sha_l+1):#length difference+1 will give the number of iterations required to shift the moving windown from start to end of the LC (with a difference of one point, two window positions are required etc.)
            end_p=start_p+sha_l-1 #-1 to give the index of the last included point
            if lc[0,end_p]-lc[0,start_p] != (sha_l-1)*time_res:
                continue
            skip=False#for "early abandon"
            sha_dist=0 #distance between shapelet and LC subsegment
            for i in range(sha_l):
                sha_dist += (lc[1,i+start_p]-shapelet[i])**2
                if sha_dist>=best_dist: 
                    skip=True#"early abandon"
                    break#break out of the distance calculation and skip the position of the moving window
            if skip ==False:
                best_dist=sha_dist
        return (best_dist)
    
def best_split_point(distances, set_entropy):
    """find a threshold distance that splits the provided set in a way that produces the best possible information gain, i.e. two subsets that are the most homogenous. distances is a list of tuples with three items; time-series id, distance from the tested shapelet, and the time-series' classification, where 1 and 0 are belong and other classes relative to the donor time-series. Set_entropy is the entropy of the set before splitting.
    """
    distances.sort(key=itemgetter(1))#ascending sort of distance values 
    best_gain_split=0
    best_split=0
    for distance in range(len(distances)-1):
        split_point=(distances[distance][1] + distances[distance+1][1])/2
        gain=information_gain(distances, set_entropy, split_point)
        if isinstance(gain, str) == True:
            continue
        if gain>best_gain_split:
            best_gain_split=gain
            best_split=split_point #split point that makes subsets with the smallest entropy
    return best_split

def entropy_pruning(best_gain, distances, best_split, belong_class_count, other_class_count, set_entropy):
    """Calculate information gain based on the incomplete set of distances and the assumption of a best case scenario for the remaining distances (all remaining belong_class objects below the threshold and other_class above the threshold). If the best case scenario achieves a larger information gain than the best value found so far, then False is returned and the shapelet is not pruned; distance calculations continue. Otherwise True is returned, indicating that the shapelet should be abandoned. best_gain is the best so far value of information gain, distances is a list of tuples that contain the time series id, distance to the shapelet and classification (1/0, belong/other), best_split is the split distance/threshold, belong_class_count and other_class_count are the numbers of objects in the entire set with the relevant classification, and set_entropy is the entropy of the set before splitting, required for the information gain calculation.
    """
    calc_belong=sum([lc[2] for lc in distances])
    calc_other=len(distances)-calc_belong
    distances_bcs=deepcopy(distances) #best case scenario when all the distances are included
    distances_bcs.sort(key=itemgetter(1))
    maxdist=distances_bcs[-1][1]+1
    for add_belong in range(belong_class_count-calc_belong):
        distances_bcs.append((-1,0,1))
    for add_other in range(other_class_count-calc_other):
        distances_bcs.append((-1,maxdist,0))
    gain_bcs=information_gain(distances_bcs, set_entropy, best_split)
    if isinstance(gain_bcs, str) == True:
        return True
    else:
        if gain_bcs<=best_gain:
            return True
        else:
            return False

def import_labels(file_name, id_extension):
    """load the classified observation ids and their states (from the file provided by Huppenkothen et al. 2017)
    """
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
            ob_state[ob+id_extension] = state
    return ob_state
    #xp10408013100_lc.txt classified as chi1 and chi4, xp20402011900_lc.txt as chi2 and chi2
    #del ob_state["10408-01-31-00{}".format(extension)] as long as training and test sets are checked for duplicates when appending, it should be ok to keep
    
def scaling(data, method, no_sigma=5, center="minimum"):
    """ Normalise or standardise the y-values of time series.
    method =    "normal" for normalisation y_i_norm = (y_i - y_center)/(y_max - y_min), where y_center is either y_mean or y_min as dictated                    by center argument
                "standard" for standardisation y_i_stand = (y_i - y_mean)/y_std
    no_sigma = the value of sigma to be assumed as the maximum value of y (to truncate the outliers).
    center =    "minimum" for min-max normalisation
                "mean" for mean normalisation
    """
    data_dims = np.shape(data[0])[0]
    all_counts=[]
    if data_dims == 2:
        for lc in data:
            all_counts.append(lc[1])
    else:
        all_counts=data
    all_counts_ar=np.concatenate(all_counts, axis=0)
    armean=np.mean(all_counts_ar)
    arstd=np.std(all_counts_ar)
    armedian=np.median(all_counts_ar)
    armin=np.min(all_counts_ar)
    armax=armean+no_sigma*arstd
    
    lcs_std=[]
    if method == "normal":
        if center == "minimum":
            center=armin
        elif center == "mean":
            center=armean
        else:
            print("{} is not a valid center".format(center))
            return
        if data_dims == 2:
            for ts in data:
                lc=np.copy(ts)
                lc[1]=(lc[1]-center)/(armax-armin)
                over_max=np.where(lc[1]>1.)[0]
                lc[1][over_max]=1.
                lcs_std.append(lc)
        else:
            for ts in data:
                lc=np.copy(ts)
                lc=(lc-center)/(armax-armin)
                over_max=np.where(lc>1.)[0]
                lc[over_max]=1.
                lcs_std.append(lc)
        return lcs_std
    
    elif method == "standard":
        if data_dims == 2:
            for ts in data:
                lc=np.copy(ts)
                lc[1]=(lc[1]-armean)/arstd
                lcs_std.append(lc)
        else:
            for ts in data:
                lc=np.copy(ts)
                lc=(lc-armean)/arstd
                lcs_std.append(lc)
        return lcs_std
    
    else:
        print("{} is not a valid method".format(method))
        return