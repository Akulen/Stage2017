from dt               import DT
from forest           import Forest, ParallelForest
from nn               import NN
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import tensorflow as tf
import utils

class DN1(ParallelForest):
    def __init__(self, nbInputs, maxProf, layerSize, sess=None, debug=False,
            useEt=False, nbJobs=8, nbIter=-1, pref=""):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref)
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
                debug=self.debug, useEt=self.useEt, nbIter=1, pref=id)



class DN2(Forest):
    def __init__(self, nbInputs, maxProf, layerSize, sess=None, debug=False,
            useEt=False, nbIter=-1, pref=""):
        super().__init__(nbIter=nbIter, pref=pref)
        self.pref = "decision-network-" + self.pref

        self.nbInputs    = nbInputs
        self.maxProf     = maxProf
        self.layerSize   = layerSize
        self.layers      = [[None] * self.nbIter for _ in range(3)]
        self.sess        = sess
        self.debug       = debug
        self.useEt       = useEt
        self.maxFeatures = (self.nbInputs+2) // 3

        if useEt:
            self.et = DT(self.sess, self.run, self.nbInputs, self.maxProf,
                    self.maxFeatures, nbIter=self.nbIter,
                    treeType=ExtraTreesRegressor)

        self.initSolvers()
        self.rf    = self.iters[:]
        self.iters = [None, None]

    def createSolver(self, id):
        return DT(id, self.run, self.nbInputs, self.maxProf, self.maxFeatures)

    def train(self, data, validation, nbEpochs=100, batchSize=32,
            logEpochs=False):
        #nbEpochs //= 2
        if self.useEt:
            self.et.train(data, validation, nbEpochs=nbEpochs)
        else:
            for i in range(self.nbIter):
                batch = utils.selectBatch(data, len(data)//3, replace=False,
                        unzip=False)
                self.rf[i].train(batch, validation, nbEpochs)

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

        alpha = 0.1
        activation = [
            lambda x : (tf.tanh(x) * (1-2*alpha) + 1) / 2,
            #lambda x : -1 * tf.nn.relu(-x),
            #lambda x : tf.log(tf.tanh(x) / 2.5 + 0.5),
            #lambda x : tf.log(tf.tanh(x) / 2 + 0.5),
            #lambda x : tf.tanh(x+1) - 1,
            tf.log,
            tf.nn.softmax
        ]

        #self.debug=True
        self.iters[0] = NN(self.pref, self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, bias=bias,
                activation=activation, fix=[False, True, [False,True], False],
                positiveWeight=True, sess=self.sess, debug=self.debug)

        fns = self.iters[0].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs)
        if logEpochs:
            fns = fns[:-1]

        ##Extract tree from nn
        for j in range(4):
            for i in range(self.nbIter):
                weight[j][i] = self.iters[0].getWeights(j, i)
                bias[j][i]   = self.iters[0].getBias(j, i)
        for i in range(self.nbIter):
            p = (np.tanh(gamma * (2 * np.array(weight[2][i]) - 1)) + 1) / 2
            #p = np.array(weight[2][i])
            #p = [[max(q[j][k], q[j+self.layerSize-1][k]) for k in range(self.layerSize)]
            #    for j in range(self.layerSize-1)]
            weight[2][i] = np.zeros(weight[2][i].shape)
            t = utils.buildTree(p,
                    [j for j in range(self.layers[0][i][0])],
                    [j for j in range(self.layers[2][i][0])])
            def fill(t, path=[]):
                if len(t) == 1:
                    for n in path:
                        weight[2][i][n][t[0]] = 1.
                    return
                fill(t[1], path+[t[0]])
                fill(t[2], path+[t[0]+self.layers[0][i][0]])
            fill(t)

        ##Train tree as nn
        self.iters[1] = NN(self.pref, self.run, self.nbInputs, self.layers,
                connectivity=connectivity, weight=weight, bias=bias,
                activation=activation, fix=[False, True, True, False],
                debug=self.debug, sess=self.sess)

        fns2 = self.iters[1].train(data, validation, nbEpochs=nbEpochs,
                batchSize=batchSize, logEpochs=logEpochs)

        self.best = 1
        if self.iters[0].evaluate(validation) < self.iters[1].evaluate(validation):
            self.best = 0

        if logEpochs:
            fns += fns2
            fns = fns[::2]
        return fns

    def evaluate(self, data):
        return self.iters[self.best].evaluate(data)

    def solve(self, x):
        return self.iters[self.best].solve(x)



