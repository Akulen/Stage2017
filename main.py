from dn               import DN1, DN2
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
import multiprocessing as mp

def evaluateSolver(solver, data):
    random.shuffle(data)
    n              = len(data)
    trainData      = data[:n//2]
    validationData = data[n//2:3*n//4]
    testData       = data[3*n//4:]

    solver.train(trainData, validationData)
    eval = solver.evaluate(testData)
    return eval

datafiles = [("autoMPG",     7  ),
             ("housing",     13 ),
             ("communities", 101),
             ("forest",      10 ),
             ("wisconsin",   32 ),
             ("concrete",    8  ),
             ("friedman1",   10 ),
             ("hwang5n",     2  )]

solvers = [lambda id, n, sess : NNF2(n, nbNeurones, sess=sess, debug=debug,
                nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : RF(n, nbNeurones, nbIter=nbIter,
                pref=str(id)),
           lambda id, n, sess : RNF1(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : RNF2(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : RNF1(n, maxProf, nbNeurones, sess=sess,
                sparse=False, debug=debug, nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : RNF2(n, maxProf, nbNeurones, sess=sess,
                sparse=False, debug=debug, nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : DN1(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id)),
           lambda id, n, sess : DN2(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id))]
assert len(sys.argv) > 1
iSolver = int(sys.argv[1])
assert iSolver < len(solvers)

maxProf    = 6
nbNeurones = 2**maxProf
nbIter     = 5
debug      = False

def createSolver(id, nbInputs, sess):
    solver = solvers[iSolver](id, nbInputs, sess)
    return solver

def thread(id, filename, nbInputs):
    with Session() as sess:
        with createSolver(id, nbInputs, sess) as solver:
            res = evaluateSolver(solver, getData(filename, nbInputs, 1))
    return res

if __name__ == '__main__':
    for i, (filename, nbInputs) in enumerate(datafiles[:1]):
        print("## %-15s" % filename, end='')

        def foo():
            rmse = Parallel(n_jobs=4)(
                delayed(thread)(filename + "-" + str(j), filename, nbInputs) for j in range(3)
            )

            print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
            print("%5.2f)" % np.std(rmse))

        proc = mp.Process(target=foo)
        proc.start()
        proc.join()

