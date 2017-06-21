from dt import DT
from forest import Forest
from joblib import Parallel, delayed
from math import sqrt
from nn import NN
from solver import Solver
import numpy as np
import random
import tensorflow as tf
import utils

class RNF(Forest):
    def __init__(self, nbInputs, maxProf, nbFeatures, nbIter=-1):
        super().__init__(nbIter)
        self.nbInputs = nbInputs
        self.maxProf = maxProf
        self.nbFeatures = nbFeatures
        self.layers = [
            (self.nbIter * nbFeatures, 100),
            (self.nbIter * nbFeatures, 1)
        ]
        self.connectivity = [
            np.ones((nbInputs, self.layers[0][0])),
            np.kron(np.eye(nbIter), np.ones((nbFeatures, nbFeatures)))
        ]

        self.dt = [DT(id, self.run, self.nbInputs, self.nbInputs//3, self.maxProf) for _ in range(self.nbIter)]

    def train(self, data, validation, nbEpochs=100):
        for i in range(self.nbIter):
            batch = utils.selectBatch(data, len(data)//3, replace=False, unzip=False)
            self.dt[i].train(batch, validation, nbEpochs)

        self.connectivity[0] = np.zeros((self.nbInputs, self.layers[0][0]))
        weight = [
            np.zeros((self.nbInputs, self.nbIter * self.nbFeatures)),
            np.zeros((self.nbIter * self.nbFeatures, self.nbIter * self.nbFeatures)),
            np.zeros((self.nbIter * self.nbFeatures, 1))
        ]
        bias = [
            np.zeros(self.nbIter * self.nbFeatures),
            np.zeros(self.nbIter * self.nbFeatures),
            np.array([sum([sum([v[0][0] for v in dt.tree.tree_.value]) / 2
                for dt in self.dt]) / self.nbIter])
        ]
        for i in range(self.nbIter):
            father = [-1] * self.dt[i].tree.tree_.node_count
            value = [0] * self.dt[i].tree.tree_.node_count
            for j in range(self.dt[i].tree.tree_.node_count):
                if self.dt[i].tree.tree_.children_left[j] >= 0:
                    father[self.dt[i].tree.tree_.children_left[j]] = j
                    father[self.dt[i].tree.tree_.children_right[j]] = j
                    value[self.dt[i].tree.tree_.children_left[j]] = 1.
                    value[self.dt[i].tree.tree_.children_right[j]] = -1.
            for j in range(self.dt[i].tree.tree_.node_count):
                weight[2][i * self.nbFeatures + j][0] = self.dt[i].tree.tree_.value[j] / 2
                if self.dt[i].tree.tree_.feature[j] < 0:
                    v = value[j]
                    cur = father[j]
                    l = 0
                    while cur != -1:
                        self.connectivity[1][i*self.nbFeatures+cur][i*self.nbFeatures+j] = 1.
                        weight[1][i*self.nbFeatures+cur][i*self.nbFeatures+j] = v
                        l += 1
                        v = value[cur]
                        cur = father[cur]
                    bias[1][i * self.nbFeatures + j] = 0.5 - l
                else:
                    self.connectivity[0][self.dt[i].tree.tree_.feature[j]][i*self.nbFeatures+j] = 1.
                    weight[0][self.dt[i].tree.tree_.feature[j]][i * self.nbFeatures + j] = 1.
                    bias[0][i * self.nbFeatures + j] = - self.dt[i].tree.tree_.threshold[j]

        self.nn = NN(id, self.run, self.nbInputs, self.layers, connectivity=self.connectivity, weight=weight)

        self.nn.train(data, validation, nbEpochs)

    def evaluate(self, data):
        z = [None] * len(data)
        for j in range(len(data)):
            x, y = data[j]
            z[j] = self.nn.solve([x])
        return utils.evaluate(z, [y[0] for _, y in data])





