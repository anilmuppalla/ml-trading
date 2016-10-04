"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rtl
import BagLearner as bl
import sys
import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size":1}, bags = 20, boost = False, verbose = False)
    # learner.addEvidence(trainX, trainY) # train it

    df = pd.DataFrame(columns = ['Bag Size', 'Train', 'Test'])

    for i in range(1,51):
        print i

        rmse1 = 0
        rmse2 = 0

        learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size":5}, bags = i, boost = False, verbose = False)

        for j in range(10):
            #learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size":20}, bags = i, boost = False, verbose = False)
            learner.addEvidence(trainX, trainY)
            predY = learner.query(trainX)
            rmse1 += math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            #rmse1 += np.corrcoef(predY, y=trainY)[0,1]
            predY = learner.query(testX)
            rmse2 += math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            #rmse2 += np.corrcoef(predY, y=testY)[0,1]
        rmse1 /=10.0
        rmse2 /=10.0
        #cor1 = np.corrcoef(predY, y=trainY)[0,1]
        #predY = learner.query(testX)
        #cor2 = np.corrcoef(predY, y=testY)[0,1]
        df.loc[i] = np.array([i,rmse1,rmse2])

    # df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
    line_train = plt.plot(df.index.values,df['Train'], label = 'Train')
    line_test = plt.plot(df.index.values,df['Test'], label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Bag Size')
    plt.xticks(rotation=30)
    plt.ylabel('RMSE')
    plt.title('BagLearner: RMSE vs Bag Size (Leaf Size 5)')
    plt.show()

    # # evaluate in sample
    # predY = learner.query(trainX) # get the predictions
    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    # print
    # print "In sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=trainY)
    # print "corr: ", c[0,1]

    # # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # accuracy = 100.0*testY[testY==predY].shape[0]/testY.shape[0]
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]
    # print
    # print "ACCURACY:", accuracy
