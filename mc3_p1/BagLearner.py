import numpy as np
import LinRegLearner as lr
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.learners = []
        self.bags = bags
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        data = np.concatenate((dataX, dataY[:, None]), axis = 1)
        cols = dataX.shape[1]
        rows = data.shape[0]
        dataset = [[0] * rows for i in range(self.bags)]

        for i in range(self.bags):
            dataset[i] = data[np.random.choice(np.arange(0,rows -1), size = rows),:]

        for i in range(self.bags):
            datasetX = np.hsplit(dataset[i],[cols])[0]
            datasetY = np.hsplit(dataset[i],[cols])[1].flatten()
            self.learners[i].addEvidence(datasetX, datasetY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        predictY = np.empty(shape=(points.shape[0],))
        for i in range(0,self.bags):
            predictY += self.learners[i].query(points)
        predictY = predictY/self.bags
        return predictY

if __name__=="__main__":
    print "LULZ"