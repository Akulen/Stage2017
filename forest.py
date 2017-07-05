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

    def train(self, data, validation, nbEpochs=100, batchSize=32,
            logEpochs=False):
        r = [it.train(data, validation, nbEpochs, logEpochs=logEpochs)
                for it in self.iters]
        if logEpochs:
            fns = [[0 for _ in range(len(r[0][0]))] for _ in range(len(r[0]))]

            for f in r:
                for t in range(len(f)):
                    for x in range(len(f[t])):
                        fns[t][x] += f[t][x] / len(r)

            return fns

    def evaluate(self, data):
        z = []
        for x, y in data:
            res = [it.solve([x])[0] for it in self.iters]
            z.append(sum(res) / self.nbIter)
        return utils.evaluate(z, [y[0] for _, y in data])

    def close(self):
        for it in self.iters:
            it.close()

class ParallelForest(Forest):
    __metaclass__ = ABCMeta

    def __init__(self, nbIter=-1, nbJobs=8, pref=""):
        super().__init__(nbIter, "parallel-" + pref)
        self.nbJobs     = nbJobs

        for i in range(self.nbIter):
            self.iters[i] = self.createSolver(self.pref + str(i))

    def train(self, data, validation, nbEpochs=100, logEpochs=False):
        #r = Parallel(n_jobs=self.nbJobs)(
        #    delayed(self.iters[i].train)(data, validation, nbEpochs, logEpochs)
        #        for i in range(self.nbIter)
        #)
        return super().train(data, validation, nbEpochs=nbEpochs,
                logEpochs=logEpochs)

    @abstractmethod
    def createSolver(self, id):
        pass

    def evaluate(self, data):
        x = [_x for _x, _ in data]
        z = self.solve(x)
        return utils.evaluate(z, [y[0] for _, y in data])

    def solve(self, data):
        z = [0] * len(data)
        for j in range(len(data)):
            x = data[j]
            #r = Parallel(n_jobs=self.nbJobs)(
            #    delayed(self.iters[i].solve)([x]) for i in range(self.nbIter)
            #)
            r = [self.iters[i].solve([x]) for i in range(self.nbIter)]
            z[j] = sum([_r[0][0] for _r in r]) / self.nbIter
        return z





