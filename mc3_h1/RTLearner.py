"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.random_tree = self.build_random_tree(dataX,dataY)

    def build_random_tree(self, dataX, dataY):
        #factor, splitval, left, right
        # if dataX.shape[0] == 0:
        #     return np.array([[-1,-1,-1,-1]])
        if dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(dataY), -1, -1]])
        if len(np.unique(dataY)) == 1: 
            return np.array([[-1, dataY[0], -1, -1]])

        leftTreeSize = rightTreeSize = 0

        while leftTreeSize == 0 or rightTreeSize == 0:
            randomFeature = np.random.randint(0,dataX.shape[1])
            randomFeatureRow1 = np.random.randint(0,dataX.shape[0])
            randomFeatureRow2 = np.random.randint(0,dataX.shape[0])
            splitVal = (dataX[randomFeatureRow1, randomFeature] + dataX[randomFeatureRow2,randomFeature])/float(2)
            leftTreeSize = dataX[dataX[:,randomFeature]<=splitVal].shape[0]
            rightTreeSize = dataX[dataX[:,randomFeature]>splitVal].shape[0]
        
        leftTree = self.build_random_tree(dataX[dataX[:,randomFeature]<=splitVal], dataY[dataX[:,randomFeature]<=splitVal])
            
        rightTree = self.build_random_tree(dataX[dataX[:,randomFeature]>splitVal], dataY[dataX[:,randomFeature] > splitVal])

        root = [randomFeature, splitVal, 1, leftTree.shape[0]+1]
        
        return np.vstack((root,leftTree,rightTree))

    def search(self,point,index):
        node = self.random_tree[index]
        if node.item(0) == -1:
            return node.item(1)
        elif point[int(node.item(0))] <= node.item(1):
            return self.search(point, int(node.item(2)) + index)
        else:
            return self.search(point, int(node.item(3)) + index)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        predictY = np.empty(shape=(points.shape[0],))
        for i in range(0,len(points)):
            predictY[i] = self.search(points[i],0)
        return predictY

if __name__=="__main__":
    print "LULZ!"
