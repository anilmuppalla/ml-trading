"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg():
    X = np.random.normal(size = (250, 4)) 
    Y = 0.45 * X[:,0] + 0.4 * X[:,1] - 0.2 * X[:,2] + 0.6 * X[:,3] + 0.5
    return X, Y

def best4RT():
    X = np.random.choice(1, size = (200, 3)) 
    Y = np.zeros(shape=(200,))
    for i in xrange(X.shape[0]):
        x = X[i, :]
        if x[0]>0 and x[1]>0 and x[2]>0:
            Y[i] = 1
        elif x[0]>0 and x[1]>0 and x[2]<=0:
            Y[i] = 2
        elif x[0]>0 and x[1]<=0 and x[2]>0:
            Y[i] = 2
        elif x[0]>0 and x[1]<=0 and x[2]<=0:
            Y[i] = 1
        elif x[0]<=0 and x[1]>0 and x[2]<=0:
            Y[i] = 1
        elif x[0]<=0 and x[1]<=0 and x[2]>0:
            Y[i] = 2
        elif x[0]<=0 and x[1]<=0 and x[2]<=0:
            Y[i] = 1
        elif x[0]<=0 and x[1]>0 and x[2]>0:
            Y[i] = 2
    return X, Y

if __name__=="__main__":
    print "they call me Tim."
