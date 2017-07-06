from forest           import Forest
from math             import sqrt
from sklearn.ensemble import RandomForestRegressor
from solver           import Solver
import numpy as np
import random
import utils

class DT(Solver):
    def __init__(self, id, run, nbInputs, maxProf, maxFeatures, nbIter=1,
            treeType=RandomForestRegressor):
        super().__init__(id, run, nbInputs)
        self.maxProf     = maxProf
        self.maxFeatures = maxFeatures
        self.nbIter      = nbIter

        self.tree = treeType(n_estimators=nbIter, max_depth=maxProf,
                max_features=maxFeatures)

    def train(self, data, validation, nbEpochs=100):
        X, y = utils.zipData(data)
        self.tree.fit(X, y)

    def evaluate(self, data):
        xs, ys = utils.zipData(data)
        return utils.evaluate(self.tree.predict(xs), ys)

    def solve(self, x):
        return self.tree.predict(x)

    def makeTree(self, id=0):
        tree   = self.tree.estimators_[id].tree_
        father = [-1] * tree.node_count
        side   = [0] * tree.node_count
        for i in range(tree.node_count):
            if tree.children_left[i] >= 0:
                father[tree.children_left[i]]  = i
                father[tree.children_right[i]] = i
                side[tree.children_left[i]]    = -1.
                side[tree.children_right[i]]   = 1.
        return father, side

    def indexNodes(self, id=0):
        tree  = self.tree.estimators_[id].tree_
        nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] >= 0:
                nodes.append(i)
        return nodes

    def indexLeafs(self, id=0):
        tree = self.tree.estimators_[id].tree_
        leafs = []
        for i in range(tree.node_count):
            if tree.children_left[i] < 0:
                leafs.append(i)
        return leafs

class RF(Forest):
    def __init__(self, nbInputs, maxProf, nbIter=-1, pref=""):
        super().__init__(nbIter, pref)
        self.pref = "random-" + self.pref

        self.nbInputs    = nbInputs
        self.maxProf     = maxProf
        self.maxFeatures = (nbInputs + 2) // 3

        self.initSolvers()

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures)

    def train(self, data, validation, nbEpochs=100, logEpochs=False):
        for it in self.iters:
            batch = utils.selectBatch(data, len(data)//3, replace=False,
                    unzip=False)
            it.train(batch, validation, nbEpochs=nbEpochs)

        if logEpochs:
            fn = [[self.solve(x)
                for x in np.linspace(0, 1, 10**3)]] * (nbEpochs+1)
            return fn

    def solve(self, x):
        if isinstance(x[0], list):
            return [self.solve(_x) for _x in x]
        res = [it.solve([x])[0] for it in self.iters]
        z = sum(res) / self.nbIter
        return z





