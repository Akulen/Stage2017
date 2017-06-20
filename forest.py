import utils
from solver import Solver
from math import sqrt

def square(x):
    return x*x

class Forest(object):
    def __init__(self, nbIter=-1):
        if nbIter == -1:
            nbIter = 30
        self.nbIter = nbIter
        self.run = utils.custom_iso()
        self.iters = [Solver(i, self.run, 0) for i in range(self.nbIter)]

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        for it in self.iters:
            it.train(data, validation, nbEpochs, batchSize)

    def evaluate(self, data):
        z = []
        for x, y in data:
            res = [it.solve([x])[0] for it in self.iters]
            z.append(sum(res) / self.nbIter)
        return utils.evaluate(z, [y[0] for _, y in data])
