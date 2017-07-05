from dt               import RF
from joblib           import Parallel, delayed
from math             import sqrt
from nn               import NNF1, NNF2
from rnf              import RNF1, RNF2
from sklearn.ensemble import RandomForestRegressor
from tensorflow       import Session
from utils            import getData
import numpy as np
import random
import sys

def evaluateSolver(solver, data):
    random.shuffle(data)
    n              = len(data)
    trainData      = data[:n//2]
    validationData = data[n//2:3*n//4]
    testData       = data[3*n//4:]

    solver.train(trainData, validationData)
    eval = solver.evaluate(testData)
    solver.close()
    return eval

datafiles = [("autoMPG",     7  ),
             ("housing",     13 ),
             ("communities", 101),
             ("forest",      10 ),
             ("wisconsin",   32 ),
             ("concrete",    8  )]

solvers = [lambda id, n, sess : NNF2(n, nbNeurones, nbIter, sess=sess,
                pref=str(id)),
           lambda id, n, sess : RF(n, nbNeurones, nbIter, pref=str(id)),
           lambda id, n, sess : RNF1(n, maxProf, nbNeurones, nbIter, sess=sess,
                pref=str(id)),
           lambda id, n, sess : RNF2(n, maxProf, nbNeurones, nbIter, sess=sess,
                pref=str(id)),
           lambda id, n, sess : RNF1(n, maxProf, nbNeurones, nbIter,
                sparse=False, sess=sess, pref=str(id)),
           lambda id, n, sess : RNF2(n, maxProf, nbNeurones, nbIter,
                sparse=False, sess=sess, pref=str(id))]
assert len(sys.argv) > 1
iSolver = int(sys.argv[1])
assert iSolver < len(solvers)

maxProf    = 6
nbNeurones = 2**maxProf
nbIter     = 10

def createSolver(id, nbInputs, sess):
    solver = solvers[iSolver](id, nbInputs, sess)
    return solver

def thread(id, filename, nbInputs):
    sess = Session()
    solver = createSolver(id, nbInputs, sess)
    res = evaluateSolver(solver, getData(filename, nbInputs, 1))
    solver.close()
    return res

if __name__ == '__main__':
    for i, (filename, nbInputs) in enumerate(datafiles):
        print("## " + filename)

        rmse = Parallel(n_jobs=1)(
            delayed(thread)(j, filename, nbInputs) for j in range(30)
        )

        print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
        print("%5.2f)" % np.std(rmse))

