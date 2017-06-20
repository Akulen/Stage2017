import random
import utils
from forest import Forest
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from solver import Solver

class DT(Solver):
    def __init__(self, id, run, nbInputs, maxFeatures, maxProf):
        super().__init__(id, run, nbInputs)
        self.maxProf = maxProf
        self.maxFeatures = maxFeatures

        self.tree = DecisionTreeRegressor(max_features=maxFeatures, max_depth=maxProf)

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        X, y = map(list, zip(* data))
        y = [yy[0] for yy in y]
        self.tree.fit(X, y)

    def evaluate(self, data):
        xs, ys = map(list, zip(* data))
        ys = [yy[0] for yy in ys]
        return utils.evaluate(self.tree.predict(xs), ys)

    def solve(self, x):
        return self.tree.predict(x)

class RF(Forest):
    def __init__(self, nbInputs, maxProf, nbIter=-1):
        super().__init__(nbIter)
        maxFeatures = nbInputs // 3
        self.nbInputs = nbInputs
        self.maxProf = maxProf
        for i in range(self.nbIter):
            self.iters[i] = DT(i, self.run, self.nbInputs, maxFeatures, self.maxProf)

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        for it in self.iters:
            batch = utils.selectBatch(data, len(data)//3, replace=False, unzip=False)
            it.train(batch, validation, nbEpochs, batchSize)





