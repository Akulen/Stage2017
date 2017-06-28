from dt               import DT
from forest           import Forest,   ParallelForest
from joblib           import Parallel, delayed
from math             import sqrt
from nn               import NN
from sklearn.ensemble import ExtraTreesRegressor
from solver           import Solver
import numpy      as np
import random
import tensorflow as tf
import utils

class RNF1(ParallelForest):
    def __init__(self, nbInputs, maxProf, nbFeatures, nbIter=-1, sparse=True,
            sess=None, pref=""):
        pp = "sparse-" if sparse else ""
        super().__init__(nbIter, pref=pp + "random-neural-" + pref)
        self.nbInputs   = nbInputs
        self.maxProf    = maxProf
        self.nbFeatures = nbFeatures
        self.sparse     = sparse
        self.sess       = sess

    def createSolver(self, id):
        return RNF2(self.nbInputs, self.maxProf, self.nbFeatures, 1,
                self.sparse, id, sess=self.sess)



class RNF2(Forest):
    def __init__(self, nbInputs, maxProf, nbFeatures, nbIter=-1, sparse=True,
            id=0, sess=None, pref=""):
        pp = "sparse-" if sparse else ""
        super().__init__(nbIter, pref=pp + "random-neural-" + pref)
        self.id         = id
        self.nbInputs   = nbInputs
        self.maxProf    = maxProf
        self.nbFeatures = nbFeatures
        self.sparse     = sparse
        self.sess       = sess
        self.layers     = [
            (nbFeatures-1, 100),
            (nbFeatures,   1  )
        ]

        self.dt = [DT(i, self.run, self.nbInputs, (self.nbInputs+2)//3, self.maxProf)
                for i in range(self.nbIter)]
        self.et = ExtraTreesRegressor(n_estimators=self.nbIter)

    def train(self, data, validation, nbEpochs=100, use_et=False):
        if not use_et:
            for i in range(self.nbIter):
                batch = utils.selectBatch(data, len(data)//3, replace=False, unzip=False)
                self.dt[i].train(batch, validation, nbEpochs)
        else:
            x, y = map(np.array, zip(* data))
            y = [yy[0] for yy in y]
            self.et.fit(x.reshape(-1, 1), y)

        connectivity = [[] for _ in range(2)]
        weight       = [[] for _ in range(3)]
        bias         = [[] for _ in range(3)]

        for i in range(self.nbIter):
            self.dt[i].tree = self.et.estimators_[i]
            c, w, b = utils.dt2nn(self.dt[i], self.nbInputs, self.layers[0][0],
                    self.layers[1][0], self.nbIter)

            for j in range(3):
                if j < 2:
                    connectivity[j].append(c[j])
                weight[j].append(w[j])
                bias[j].append(b[j])

        if not self.sparse:
            for i in range(len(connectivity)):
                for j in range(len(connectivity[i])):
                    connectivity[i][j] = np.ones(connectivity[i][j].shape)

        self.nn = NN(self.id, self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, bias=bias,
                sess=self.sess, pref=self.pref)

        self.nn.train(data, validation, nbEpochs)

    def evaluate(self, data):
        return self.nn.evaluate(data)

    def solve(self, x):
        return self.nn.solve(x)





