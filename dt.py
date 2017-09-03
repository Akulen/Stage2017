from forest           import Forest
from math             import sqrt
from numpy.random     import RandomState
from sklearn.ensemble import RandomForestRegressor
from solver           import Solver
import numpy as np
import random
import utils

class DT(Solver):
    def __init__(self, id, run, nbInputs, maxProf, maxFeatures, nbIter=1,
            treeType=RandomForestRegressor, rs=None):
        super().__init__(id, run, nbInputs, rs=rs)
        self.maxProf     = maxProf
        self.maxFeatures = maxFeatures
        self.nbIter      = nbIter

        self.tree = treeType(n_estimators=nbIter, max_depth=maxProf,
                max_features=maxFeatures, random_state=RandomState(self.rs.randint(1E9)))

    def train(self, data, validation, nbEpochs=100, logEpochs=None, sampler=None):
        X, y = utils.zipData(data)
        self.tree.fit(X, y)

        if logEpochs:
            fn = [[self.solve([x]) for x in logEpochs]] * (nbEpochs+1)
            return fn

    def evaluate(self, data):
        xs, ys = utils.zipData(data)
        return utils.evaluate(self.tree.predict(xs), ys)

    def solve(self, x):
        return self.tree.predict(x)

class RF(Forest):
    def __init__(self, nbInputs, maxProf, complete=False, nbIter=-1, pref="",
            rs=None):
        super().__init__(nbIter, pref, rs=rs)
        self.pref = "random-" + self.pref

        self.nbInputs    = nbInputs
        self.maxProf     = maxProf
        self.maxFeatures = nbInputs if complete else (nbInputs + 2) // 3
        self.complete    = complete

        self.initSolvers()

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures,
                rs=RandomState(self.rs.randint(1E9)))

    def train(self, data, validation, nbEpochs=100, logEpochs=None, sampler=None):
        for it in self.iters:
            rss = RandomState(sampler.randint(1E9))
            if self.complete:
                batch = utils.selectBatch(data, len(data), replace=True,
                        unzip=False, rs=rss)
            else:
                batch = utils.selectBatch(data, len(data)//3, replace=False,
                        unzip=False, rs=rss)
            it.train(batch, validation, nbEpochs=nbEpochs)

        if logEpochs:
            fn = [[self.solve(x) for x in logEpochs]] * (nbEpochs+1)
            return fn

    def solve(self, x):
        if isinstance(x[0], list):
            return [self.solve(_x) for _x in x]
        res = [it.solve([x])[0] for it in self.iters]
        z = sum(res) / self.nbIter
        return z





