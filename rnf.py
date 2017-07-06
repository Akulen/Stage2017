from dt               import DT
from forest           import Forest, ParallelForest
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
    def __init__(self, nbInputs, maxProf, layerSize, useRelu=False,
            fixMiddle=False, sess=None, debug=False, sparse=True, useEt=False,
            nbJobs=8, nbIter=-1, pref=""):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref)
        self.pref = ("sparse-" if sparse else "") + "random-neural-" + self.pref 

        self.nbInputs  = nbInputs
        self.maxProf   = maxProf
        self.layerSize = layerSize
        self.useRelu   = useRelu
        self.fixMiddle = fixMiddle
        self.sess      = sess
        self.debug     = debug
        self.sparse    = sparse
        self.useEt     = useEt

        self.initSolvers()

    def createSolver(self, id):
        return RNF2(self.nbInputs, self.maxProf, self.layerSize,
                useRelu=self.useRelu, fixMiddle=self.fixMiddle, sess=self.sess,
                debug=self.debug, sparse=self.sparse, useEt=self.useEt,
                nbIter=1, pref=self.pref + "-" + str(id))



class RNF2(Forest):
    def __init__(self, nbInputs, maxProf, layerSize, useRelu=False,
            fixMiddle=False, sess=None, debug=False, sparse=True, useEt=False,
            nbIter=-1, pref=""):
        self.nbInputs    = nbInputs
        self.maxProf     = maxProf
        self.layers      = [
            (layerSize-1, 100),
            (layerSize,     1)
        ]
        self.useRelu     = useRelu
        self.fixMiddle   = fixMiddle
        self.sess        = sess
        self.debug       = debug
        self.sparse      = sparse
        self.useEt       = useEt
        self.maxFeatures = (self.nbInputs+2) // 3

        super().__init__(nbIter=nbIter, pref=pref)
        self.pref = ("sparse-" if sparse else "") + "random-neural-" + self.pref 

        if useEt:
            self.et = DT(self.sess, self.run, self.nbInputs, self.maxProf,
                    self.maxFeatures, nbIter=self.nbIter,
                    treeType=ExtraTreesRegressor)
        else:
            self.initSolvers()
            self.rf = self.iters[:]

        self.iters = [None]

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures)

    def train(self, data, validation, nbEpochs=100, batchSize=32,
            logEpochs=False):
        if not self.useEt:
            for i in range(self.nbIter):
                batch = utils.selectBatch(data, len(data)//3, replace=False,
                        unzip=False)
                self.rf[i].train(batch, validation, nbEpochs)
        else:
            self.et.train(data, validation, nbEpochs=nbEpochs)

        connectivity = [[] for _ in range(2)]
        weight       = [[] for _ in range(3)]
        bias         = [[] for _ in range(3)]

        for i in range(self.nbIter):
            if self.useEt:
                self.rf[i].tree = self.et.tree.estimators_[i]
            c, w, b = utils.dt2nn(self.rf[i], self.nbInputs, self.layers[0][0],
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

        self.iters[0] = NN(self.pref, self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, bias=bias,
                useRelu=self.useRelu, fixMiddle=self.fixMiddle, sess=self.sess,
                debug=self.debug)

        return self.iters[0].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs)

    def evaluate(self, data):
        return self.iters[0].evaluate(data)

    def solve(self, x):
        return self.iters[0].solve(x)





