from abc    import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from math   import sqrt
from solver import Solver
import utils

class Forest(object):
    __metaclass__ = ABCMeta

    def __init__(self, nbIter=-1, pref=""):
        if nbIter == -1:
            nbIter = 30
        self.nbIter = nbIter
        self.pref   = pref + "forest"
        self.run    = utils.custom_iso()
        self.iters  = [None for _ in range(self.nbIter)]

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        for it in self.iters:
            it.train(data, validation, nbEpochs, batchSize)

    def evaluate(self, data):
        z = []
        for x, y in data:
            res = [it.solve([x])[0] for it in self.iters]
            z.append(sum(res) / self.nbIter)
        return utils.evaluate(z, [y[0] for _, y in data])

class ParallelForest(Forest):
    __metaclass__ = ABCMeta

    def __init__(self, nbIter=-1, nbJobs=8, pref=""):
        super().__init__(nbIter, "parallel-" + pref)
        self.nbJobs     = nbJobs

    def train(self, data, validation, nbEpochs=100):
        self.data       = data
        self.validation = validation
        self.nbEpochs   = nbEpochs

    @abstractmethod
    def createSolver(self, id):
        pass

    def thread(self, id, data):
        solver = self.createSolver(self.pref + str(id), self)
        solver.train(self.data, self.validation, self.nbEpochs)

        z = [None] * len(data)
        for j in range(len(data)):
            x, y = data[j]
            z[j] = solver.solve([x])[0][0]
        return z

    def evaluate(self, data):
        z = [0] * len(data)
        res = Parallel(n_jobs=self.nbJobs)(
            delayed(self.thread)(i, data) for i in range(self.nbIter)
        )

        for j in range(len(data)):
            for i in range(self.nbIter):
                z[j] += res[i][j]
            z[j] = z[j] / self.nbIter
        return utils.evaluate(z, [y[0] for _, y in data])





