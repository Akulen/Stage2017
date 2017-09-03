from dt               import DT
from forest           import Forest, ParallelForest
from math             import e
from nn               import NN
from numpy.random     import RandomState
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import tensorflow as tf
import utils

class DN1(ParallelForest):
    def __init__(self, nbInputs, maxProf, layerSize, sess=None, debug=False,
            useEt=False, nbJobs=8, nbIter=-1, pref="", rs=None):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref, rs=rs)
        self.pref = "decision-network-" + self.pref 

        self.nbInputs       = nbInputs
        self.maxProf        = maxProf
        self.layerSize      = layerSize
        self.sess           = sess
        self.debug          = debug
        self.useEt          = useEt

        self.initSolvers()

    def createSolver(self, id):
        return DN2(self.nbInputs, self.maxProf, self.layerSize, sess=self.sess,
                debug=self.debug, useEt=self.useEt, nbIter=1, pref=id, rs=self.rs)



class DN2(Forest):
    def __init__(self, nbInputs, maxProf, layerSize, sess=None, debug=False,
            useEt=False, nbIter=-1, pref="", rs=None):
        super().__init__(nbIter=nbIter, pref=pref, rs=rs)
        self.pref = "decision-network-" + self.pref

        self.nbInputs    = nbInputs
        self.maxProf     = maxProf
        self.layerSize   = layerSize
        self.layers      = [[None] * self.nbIter for _ in range(3)]
        self.sess        = sess
        self.debug       = debug
        self.useEt       = useEt
        self.maxFeatures = (self.nbInputs+2) // 3

        #self.maxProf = None
        if useEt:
            self.et = DT(self.sess, self.run, self.nbInputs, self.maxProf,
                    self.maxFeatures, nbIter=self.nbIter,
                    treeType=ExtraTreesRegressor, rs=RandomState(self.rs.randint(1E9)))

        self.initSolvers()
        self.rf    = self.iters[:]
        self.iters = [None, None]

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures,
                rs=RandomState(self.rs.randint(1E9)))

    def train(self, data, validation, nbEpochs=100, batchSize=32,
            logEpochs=None, sampler=None):
        nbEpochs //= 2
        if self.useEt:
            fff = self.et.train(data, validation, nbEpochs=nbEpochs,
                    logEpochs=logEpochs)
        else:
            for i in range(self.nbIter):
                batch = utils.selectBatch(data, len(data)//3, replace=False,
                        unzip=False, rs=sampler)
                fff = self.rf[i].train(batch, validation, nbEpochs,
                        logEpochs=logEpochs)

        connectivity = [[] for _ in range(3)]
        weight       = [[] for _ in range(4)]
        bias         = [[] for _ in range(4)]

        gamma = 100

        for i in range(self.nbIter):
            if self.useEt:
                tree = self.et.tree.estimators_[i]
            else:
                tree = self.rf[i].tree.estimators_[0]
            nbNodes = tree.tree_.node_count // 2
            self.layers[0][i] = (nbNodes,   1)
            self.layers[1][i] = (2*nbNodes, 1)
            self.layers[2][i] = (nbNodes+1, 1)
            
            c, w, b = utils.dt2dn(self.rf[i], tree.tree_, self.nbInputs,
                    self.layers[0][i][0], self.layers[1][i][0],
                    self.layers[2][i][0], self.nbIter, gamma)

            for j in range(4):
                if j < 3:
                    connectivity[j].append(c[j])
                weight[j].append(w[j])
                bias[j].append(b[j])

        self.sparse = False
        if not self.sparse:
            for i in range(len(connectivity)):
                for j in range(len(connectivity[i])):
                    connectivity[i][j] = np.ones(connectivity[i][j].shape)

        # Randomly initialize weights -- Bad results because learning stuck
        #for i in range(len(weight)):
        #    for j in range(len(weight[i])):
        #        if i != 1:
        #            weight[i][j] = None
        #            if i != 2:
        #                bias[i][j] = None

        alpha = 1 / (2 * len(data))
        activation = [
            lambda x : (tf.tanh(x) * (1-2*alpha) + 1) / 2,
            #lambda x : -1 * tf.nn.relu(-x),
            #lambda x : tf.log(tf.tanh(x) / 2.5 + 0.5),
            #lambda x : tf.log(tf.tanh(x) / 2 + 0.5),
            #lambda x : tf.tanh(x+1) - 1,
            lambda x : 1 - 1 / x,
            #tf.log,
            tf.nn.softmax
        ]
        #print(weight[2][0])

        #self.debug=True
        self.iters[0] = NN(self.pref, self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, bias=bias,
                activation=activation, fix=[False, True, [False,True], False],
                positiveWeight=True, sess=self.sess, debug=self.debug,
                rs=self.rs)

        fns = self.iters[0].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs, sampler=sampler)
        if logEpochs:
            fns = fns[:-1]

        ##Extract tree from nn
        for j in range(4):
            for i in range(self.nbIter):
                weight[j][i] = self.iters[0].getWeights(j, i)
                bias[j][i]   = self.iters[0].getBias(j, i)

        oldW = [None] * self.nbIter

        for i in range(self.nbIter):
            p       = np.array(weight[2][i])
            oldW[i] = np.array(weight[2][i])

            t = utils.buildTree(p,
                    [j for j in range(self.layers[0][i][0])],
                    [j for j in range(self.layers[2][i][0])],
                    balance=False)

            weight[2][i] = np.zeros(weight[2][i].shape)
            def fill(t, path=[]):
                if len(t) == 1:
                    for n in path:
                        weight[2][i][n][t[0]] = 1.
                    return
                fill(t[1], path+[t[0]])
                fill(t[2], path+[t[0]+self.layers[0][i][0]])
            fill(t)
        #print(weight[2][0])

        ##Train tree as nn
        self.iters[1] = NN(self.pref + "-soft-tree", self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, smoothW=oldW, bias=bias,
                activation=activation, fix=[False, True, True, False],
                debug=self.debug, sess=self.sess, rs=self.rs)

        fns2 = self.iters[1].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs, sampler=sampler)

        self.best = 0
        if self.iters[0].evaluate(validation) < self.iters[1].evaluate(validation):
            self.best = 0
        #print(nbNodes)

        if logEpochs:
            fns += fns2
            #fns = fns[::2]
            #fns[3] = np.array(fns[0])
            #fns[4] = np.array(fns[1])
            #fns[5] = np.array(fns[2])
            #treef  = [x[0] for x in fff[0]]
            #fns[0] = np.array(treef)
            #fns[1] = np.array(treef)
            #fns[2] = np.array(treef)
        return fns

    def evaluate(self, data):
        return self.iters[self.best].evaluate(data)

    def solve(self, x):
        return self.iters[self.best].solve(x)



