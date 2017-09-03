from dn               import DN1, DN2
from dt               import RF
from joblib           import Parallel, delayed
from math             import sqrt
from nn               import NNF1, NNF2
from numpy.random     import RandomState
from rnf              import RNF1, RNF2
from sklearn.ensemble import RandomForestRegressor
from tensorflow       import Session
from utils            import getData
import numpy as np
import sys
import multiprocessing as mp

def evaluateSolver(solver, data, rs):
    rs.shuffle(data)
    n              = len(data)
    trainData      = data[:n//2]
    validationData = data[n//2:3*n//4]
    testData       = data[3*n//4:]

    solver.train(trainData, validationData,
            sampler=RandomState(rs.randint(1E9)))
    eval = solver.evaluate(testData)
    return eval

rs = np.random.RandomState(42)

datafiles = [("autoMPG",     7  ),
             ("housing",     13 ),
             ("communities", 101),
             ("forest",      10 ),
             ("wisconsin",   32 ),
             ("concrete",    8  ),
             ("friedman1",   10 ),
             ("hwang5n",     2  )]

solvers = [lambda id, n, sess, rs : NNF1(n, nbNeurones, sess=sess, debug=debug,
                nbIter=nbIter, pref=str(id), rs=rs),
           lambda id, n, sess, rs : RF(n, maxProf, nbIter=nbIter,
                pref=str(id), rs=rs),
           lambda id, n, sess, rs : RNF1(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id), rs=rs),
           lambda id, n, sess, rs : RNF2(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id), rs=rs),
           lambda id, n, sess, rs : RNF1(n, maxProf, nbNeurones, sess=sess,
                sparse=False, debug=debug, nbIter=nbIter, pref=str(id),
                rs=rs),
           lambda id, n, sess, rs : RNF2(n, maxProf, nbNeurones, sess=sess,
                sparse=False, debug=debug, nbIter=nbIter, pref=str(id),
                rs=rs),
           lambda id, n, sess, rs : DN1(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id), rs=rs),
           lambda id, n, sess, rs : DN2(n, maxProf, nbNeurones, sess=sess,
                debug=debug, nbIter=nbIter, pref=str(id), rs=rs)]
assert len(sys.argv) > 1
iSolver = int(sys.argv[1])
assert iSolver < len(solvers)

maxProf    = 6
nbNeurones = 2**maxProf
nbIter     = 30
debug      = False

def createSolver(id, nbInputs, sess, rs):
    solver = solvers[iSolver](id, nbInputs, sess, rs=rs)
    return solver

def thread(id, filename, nbInputs, seed=None):
    rs = np.random.RandomState(seed)
    with Session() as sess:
        with createSolver(id, nbInputs, sess, RandomState(rs.randint(1E9))) \
                as solver:
            res = evaluateSolver(solver, getData(filename, nbInputs, 1), rs)
    return res

if __name__ == '__main__':
    for i, (filename, nbInputs) in enumerate(datafiles):
        #print("## %-15s" % filename, end='')

        def foo():
            rmse = Parallel(n_jobs=8)(
                delayed(thread)(filename + "-" + str(j), filename, nbInputs,
                    seed=rs.randint(1E9)) for j in range(10)
            )

            print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
            print("%5.2f)" % np.std(rmse))

        proc = mp.Process(target=foo)
        proc.start()
        proc.join()

