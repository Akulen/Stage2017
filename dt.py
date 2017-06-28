from forest import Forest
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from solver import Solver
import random
import utils

class DT(Solver):
    def __init__(self, id, run, nbInputs, maxFeatures, maxProf):
        super().__init__(id, run, nbInputs)
        self.maxProf = maxProf
        self.maxFeatures = maxFeatures

        self.tree = DecisionTreeRegressor(max_features=maxFeatures, max_depth=maxProf)

    def train(self, data, validation, nbEpochs=100, batchSize=-1):
        X, y = utils.zipData(data)
        self.tree.fit(X, y)

    def evaluate(self, data):
        xs, ys = utils.zipData(data)
        return utils.evaluate(self.tree.predict(xs), ys)

    def solve(self, x):
        return self.tree.predict(x)

    def makeTree(self):
        father = [-1] * self.tree.tree_.node_count
        side   = [0] * self.tree.tree_.node_count
        for i in range(self.tree.tree_.node_count):
            if self.tree.tree_.children_left[i] >= 0:
                father[self.tree.tree_.children_left[i]]  = i
                father[self.tree.tree_.children_right[i]] = i
                side[self.tree.tree_.children_left[i]]    = -1.
                side[self.tree.tree_.children_right[i]]   = 1.
        return father, side

    def indexNodes(self):
        nodes = []
        for i in range(self.tree.tree_.node_count):
            if self.tree.tree_.children_left[i] >= 0:
                nodes.append(i)
        return nodes

    def indexLeafs(self):
        leafs = []
        for i in range(self.tree.tree_.node_count):
            if self.tree.tree_.children_left[i] < 0:
                leafs.append(i)
        return leafs

class RF(Forest):
    def __init__(self, nbInputs, maxProf, nbIter=-1, pref=""):
        super().__init__(nbIter, "random-" + pref)
        self.nbInputs = nbInputs
        self.maxProf  = maxProf

        maxFeatures   = (nbInputs + 2) // 3
        for i in range(self.nbIter):
            self.iters[i] = DT(i, self.run, self.nbInputs, maxFeatures, self.maxProf)

    def train(self, data, validation, nbEpochs=100, batchSize=-1):
        for it in self.iters:
            batch = utils.selectBatch(data, len(data)//3, replace=False, unzip=False)
            it.train(batch, validation, nbEpochs, batchSize)

    def solve(self, x):
        res = [it.solve(x)[0] for it in self.iters]
        z = sum(res) / self.nbIter
        return z





