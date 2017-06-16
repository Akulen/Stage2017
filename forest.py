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
            print(it)
            it.train(data, validation, nbEpochs, batchSize)

    def evaluate(self, data):
        z = []
        sy = 0
        for x, y in data:
            sy += y[0]
            z.append(sum([it.solve([x]) for it in self.iters]) / self.nbIter)
        my = sy / len(data)
        resu = [square(z[i] - data[i][1]) for i in range(len(data))]
        rmse = sqrt(sum(resu) / len(data))
        resu = [square(z[i] - my) for i in range(len(data))]
        devi = sqrt(sum(resu) / len(data))
        return rmse, devi
