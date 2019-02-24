import numpy as np
import math
import random

def noisy(average_value):
    """Add some noise to the data"""
    return average_value+(math.cos(random.randint(0,360)*(math.pi/180))*0.1*average_value)

def generate(no_per_class, ts_len):
    """generate simple time series with unique features to test the classification algorithm
    no_per_class = the number of time series generated for every class
    ts_len = time series length, i.e. the number of evenly spaced data points in the time series"""
    lcs=[]
    classes=[]
    xs=np.arange(0,ts_len)
    for i in range(no_per_class):
        ys=[]
        peak=np.random.choice(xs[1:-1])
        for x in xs:
            y=noisy(2)
            ys.append(y)
        ys[peak]=noisy(5)
        ys[peak-1]=noisy(5)
        ys[peak+1]=noisy(5)
        lcs.append(np.stack((xs,ys)))
        classes.append("alpha")
    for i in range(no_per_class):
        ys=[]
        peak=np.random.choice(xs[:-1])
        for x in xs:
            y=noisy(2)
            ys.append(y)
        ys[peak]=noisy(10)
        lcs.append(np.stack((xs,ys)))
        classes.append("beta")
    for i in range(no_per_class):
        ys=[]
        peak=np.random.choice(xs[:-3])
        for x in xs:
            y=noisy(2)
            ys.append(y)
        ys[peak]=noisy(10)
        ys[peak+2]=noisy(5)
        ys[peak+3]=noisy(5)
        lcs.append(np.stack((xs,ys)))
        classes.append("gamma")
    for i in range(no_per_class):
        ys=[]
        for x in xs:
            y=noisy(2)
            ys.append(y)
        lcs.append(np.stack((xs,ys)))
        classes.append("delta")
    for i in range(no_per_class):
        ys=[]
        for x in xs:
            y=noisy(5)
            ys.append(y)
        dip_start=np.random.choice(range(int(ts_len/2)))
        ys=np.array(ys)
        dip_rads = np.linspace(0, np.pi, int(ts_len/2))
        dip_sin = np.sin(dip_rads)**2   
        ys[dip_start:dip_start+int(ts_len/2)]-=(np.copy(ys[dip_start:dip_start+int(ts_len/2)])-2)*dip_sin
        ys[dip_start+int(0.66*len(dip_sin))]+=noisy(3)
        lcs.append(np.stack((xs,ys)))
        classes.append("epsilon")
    for i in range(no_per_class):
        ys=[]
        for x in xs:
            y=noisy(5)
            ys.append(y)
        dip_start=np.random.choice(range(int(ts_len/2)))
        ys=np.array(ys)
        dip_rads = np.linspace(0, np.pi, int(ts_len/2))
        dip_sin = np.sin(dip_rads)**2   
        ys[dip_start:dip_start+int(ts_len/2)]-=(np.copy(ys[dip_start:dip_start+int(ts_len/2)])-2)*dip_sin
        lcs.append(np.stack((xs,ys)))
        classes.append("zeta")
    return lcs, classes