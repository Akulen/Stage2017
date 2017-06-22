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
            (nbFeatures, 100),
            (nbFeatures, 1)
        ]
        self.connectivity = [
            [],
            []
        ]

        self.dt = [DT(i, self.run, self.nbInputs, self.nbInputs//3, self.maxProf) for i in range(self.nbIter)]

    def train(self, data, validation, nbEpochs=100):
        for i in range(self.nbIter):
            batch = utils.selectBatch(data, len(data)//3, replace=False, unzip=False)
            self.dt[i].train(batch, validation, nbEpochs)

        weight = [
            [],
            [],
            []
        ]
        bias = [
            [],
            [],
            []
            #np.array([sum([sum([v[0][0] for v in dt.tree.tree_.value]) / 2
            #    for dt in self.dt]) / self.nbIter])
        ]
        for i in range(self.nbIter):
            father = [-1] * self.dt[i].tree.tree_.node_count
            value = [0] * self.dt[i].tree.tree_.node_count
            for j in range(self.dt[i].tree.tree_.node_count):
                if self.dt[i].tree.tree_.children_left[j] >= 0:
                    father[self.dt[i].tree.tree_.children_left[j]]  = j
                    father[self.dt[i].tree.tree_.children_right[j]] = j
                    value[self.dt[i].tree.tree_.children_left[j]]   = 1.
                    value[self.dt[i].tree.tree_.children_right[j]]  = -1.

            self.connectivity[0].append(np.zeros((self.nbInputs, self.layers[0][0])))
            weight[0].append(np.zeros((self.nbInputs, self.layers[0][0])))
            bias[0].append(np.zeros(self.layers[0][0]))

            self.connectivity[1].append(np.zeros((self.layers[0][0], self.layers[1][0])))
            weight[1].append(np.zeros((self.layers[0][0], self.layers[1][0])))
            bias[1].append(np.zeros(self.layers[1][0]))

            weight[2].append(np.zeros((self.layers[1][0], 1)))
            bias[2].append(np.zeros(1))

            for j in range(self.dt[i].tree.tree_.node_count):
                if self.dt[i].tree.tree_.feature[j] >= 0:
                    self.connectivity[0][i][self.dt[i].tree.tree_.feature[j]][j] = 1.
                    weight[0][i][self.dt[i].tree.tree_.feature[j]][j]            = 1.
                    bias[0][i][j] = - self.dt[i].tree.tree_.threshold[j]
                else:
                    v   = value[j]
                    cur = father[j]
                    l   = 0
                    while cur != -1:
                        self.connectivity[1][i][cur][j]  = 1.
                        weight[1][i][cur][j]             = v

                        l   += 1
                        v    = value[cur]
                        cur  = father[cur]

                    bias[1][i][j] = 0.5 - l

                    weight[2][i][j][0]  = self.dt[i].tree.tree_.value[j] / 2 / self.nbIter
                    bias[2][i][0]      += self.dt[i].tree.tree_.value[j] / 2 / self.nbIter

        self.nn = NN(0, self.run, self.nbInputs, self.layers, connectivity=self.connectivity, weight=weight, bias=bias)

        self.nn.train(data, validation, nbEpochs)

    def evaluate(self, data):
        z = [None] * len(data)
        for j in range(len(data)):
            x, y = data[j]
            z[j] = self.nn.solve([x])[0][0]
        return utils.evaluate(z, [y[0] for _, y in data])





