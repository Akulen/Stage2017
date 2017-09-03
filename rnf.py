from dt               import DT
from forest           import Forest, ParallelForest
from joblib           import Parallel, delayed
from math             import sqrt
from nn               import NN
from numpy.random     import RandomState
from sklearn.ensemble import ExtraTreesRegressor
import numpy      as np
import random
import tensorflow as tf
import utils

class RNF1(ParallelForest):
    def __init__(self, nbInputs, maxProf, layerSize, activation=None,
            fix=False, positiveWeight=False, sess=None, debug=False,
            sparse=True, useEt=False, nbJobs=8, nbIter=-1, pref="", rs=None):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref, rs=rs)
        self.pref = ("sparse-" if sparse else "") + "random-neural-" + self.pref 

        self.nbInputs       = nbInputs
        self.maxProf        = maxProf
        self.layerSize      = layerSize
        self.activation     = activation
        self.fix            = fix
        self.positiveWeight = positiveWeight
        self.sess           = sess
        self.debug          = debug
        self.sparse         = sparse
        self.useEt          = useEt

        self.initSolvers()

    def createSolver(self, id):
        return RNF2(self.nbInputs, self.maxProf, self.layerSize,
                activation=self.activation, fix=self.fix,
                positiveWeight=self.positiveWeight, sess=self.sess,
                debug=self.debug, sparse=self.sparse, useEt=self.useEt,
                nbIter=1, pref=id, rs=RandomState(self.rs.randint(1E9)))



class RNF2(Forest):
    def __init__(self, nbInputs, maxProf, layerSize, activation=None,
            fix=False, positiveWeight=False, sess=None, debug=False,
            sparse=True, useEt=False, nbIter=-1, pref="", rs=None):
        self.nbInputs       = nbInputs
        self.maxProf        = maxProf
        self.layers         = [
            (layerSize-1, 10000),
            (layerSize,     1)
        ]
        self.activation     = activation
        self.fix            = fix
        self.positiveWeight = positiveWeight
        self.sess           = sess
        self.debug          = debug
        self.sparse         = sparse
        self.useEt          = useEt
        self.maxFeatures    = (self.nbInputs+2) // 3

        super().__init__(nbIter=nbIter, pref=pref, rs=rs)
        self.pref = ("sparse-" if sparse else "") + "random-neural-" + self.pref 

        if useEt:
            self.et = DT(self.sess, self.run, self.nbInputs, self.maxProf,
                    self.maxFeatures, nbIter=self.nbIter,
                    treeType=ExtraTreesRegressor, rs=RandomState(self.rs.randint(1E9)))

        self.initSolvers()
        self.rf = self.iters[:]
        self.iters = [None]

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures,
                rs=self.rs)

    def train(self, data, validation, nbEpochs=100, batchSize=32,
            logEpochs=None, sampler=None):
        if self.useEt:
            fff = self.et.train(data, validation, nbEpochs=nbEpochs,
                    logEpochs=logEpochs, sampler=sampler)
        else:
            for i in range(self.nbIter):
                batch = utils.selectBatch(data, len(data)//3, replace=False,
                        unzip=False, rs=sampler)
                fff = self.rf[i].train(batch, validation, nbEpochs,
                        logEpochs=logEpochs)

        connectivity = [[] for _ in range(2)]
        weight       = [[] for _ in range(3)]
        bias         = [[] for _ in range(3)]

        for i in range(self.nbIter):
            if self.useEt:
                tree = self.et.tree.estimators_[i] 
            else:
                tree = self.rf[i].tree.estimators_[0]
            c, w, b = utils.dt2nn(self.rf[i], tree.tree_, self.nbInputs,
                    self.layers[0][0], self.layers[1][0], self.nbIter)

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
                activation=self.activation, fix=self.fix,
                positiveWeight=self.positiveWeight, sess=self.sess,
                debug=self.debug, rs=self.rs)

        fns = self.iters[0].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs, sampler=sampler)
        #if logEpochs:
            #fns[3] = np.array(fns[0])
            #fns[4] = np.array(fns[1])
            #fns[5] = np.array(fns[2])
            #treef  = [x[0] for x in fff[0]]
            #fns[0] = np.array(treef)
            #fns[1] = np.array(treef)
            #fns[2] = np.array(treef)
        return fns

    def evaluate(self, data):
        return self.iters[0].evaluate(data)

    def solve(self, x):
        return self.iters[0].solve(x)





