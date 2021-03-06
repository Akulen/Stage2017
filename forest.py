from abc          import ABCMeta,  abstractmethod
from joblib       import Parallel, delayed
from math         import sqrt
from numpy.random import RandomState
from solver       import Solver
import utils

class Forest(object):
    __metaclass__ = ABCMeta

    def __init__(self, nbIter=-1, pref="", rs=None):
        if nbIter == -1:
            nbIter = 30
        if pref != "":
            pref = "-" + pref
        self.nbIter = nbIter
        self.pref   = "forest" + pref
        self.run    = utils.custom_iso()
        self.rs     = rs if rs else RandomState(None)

    def initSolvers(self):
        self.iters  = [self.createSolver(self.pref + "-" + str(i))
                for i in range(self.nbIter)]

    @abstractmethod
    def createSolver(self, id):
        pass

    def train(self, data, validation, nbEpochs=100, logEpochs=False,
            sampler=None):
        r = [it.train(data, validation, nbEpochs=nbEpochs, logEpochs=logEpochs,
                sampler=RandomState(sampler.randint(1E9))) for it in self.iters]

        if logEpochs:
            fns = [[0 for _ in range(len(r[0][0]))] for _ in range(len(r[0]))]

            for f in r:
                for t in range(len(f)):
                    for x in range(len(f[t])):
                        fns[t][x] += f[t][x] / len(r)

            return fns
    
    def solve(self, data):
        z = []
        for x in data:
            res = [it.solve([x])[0][0] for it in self.iters]
            z.append(sum(res) / self.nbIter)
        return z

    def evaluate(self, data):
        x = [_x for _x, _ in data]
        y = [_y[0] for _, _y in data]
        z = self.solve(x)
        return utils.evaluate(z, y)

    def close(self):
        for it in self.iters:
            if it is not None:
                it.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()



class ParallelForest(Forest):
    __metaclass__ = ABCMeta

    def __init__(self, nbIter=-1, nbJobs=8, pref="", rs=None):
        super().__init__(nbIter, pref, rs=rs)
        self.pref = "parallel-" + self.pref

        self.nbJobs = nbJobs

    #def train(self, data, validation, nbEpochs=100, logEpochs=False):
    #    r = Parallel(n_jobs=self.nbJobs)(
    #        delayed(self.iters[i].train)(data, validation, nbEpochs, logEpochs)
    #            for i in range(self.nbIter)
    #    )

    #def solve(self, data):
    #    r = Parallel(n_jobs=self.nbJobs)(
    #        delayed(self.iters[i].solve)([x]) for i in range(self.nbIter)
    #    )





