from dt               import RF
from joblib           import Parallel, delayed
from math             import sqrt
from nn               import NNF
from rnf              import RNF1, RNF2
from sklearn.ensemble import RandomForestRegressor
from tensorflow       import Session
from utils            import getData
import numpy as np
import random

def evaluateSolver(solver, data):
    random.shuffle(data)
    n              = len(data)
    trainData      = data[:n//2]
    validationData = data[n//2:3*n//4]
    testData       = data[3*n//4:]

    solver.train(trainData, validationData)
    return solver.evaluate(testData)

datafiles = [("autoMPG",     7  ),
             ("housing",     13 ),
             ("communities", 101),
             ("forest",      10 ),
             ("wisconsin",   32 ),
             ("concrete",    8  )]

maxProf    = 6
nbNeurones = 2**maxProf
nbIter     = 30
sess       = Session()

def createSolver(id, nbInputs):
    #solver = NNF(nbInputs, nbNeurones, nbIter, sess=sess, pref=str(id))
    #solver = RF(nbInputs, nbNeurones, nbIter, pref=str(id))
    #solver = RNF1(nbInputs, maxProf, nbNeurones, nbIter, sess=sess,
    #        pref=str(id))
    #solver = RNF1(nbInputs, maxProf, nbNeurones, nbIter, sparse=False,
    #        sess=sess, pref=str(id))
    solver = RNF2(nbInputs, maxProf, nbNeurones, nbIter, sess=sess,
            pref=str(id))
    #solver = RNF2(nbInputs, maxProf, nbNeurones, nbIter, sparse=False,
    #        sess=sess, pref=str(id))
    return solver

def thread(id, filename, nbInputs):
    solver = createSolver(id, nbInputs)
    return evaluateSolver(solver, getData(filename, nbInputs, 1))

if __name__ == '__main__':
    for i, (filename, nbInputs) in enumerate(datafiles):
        print("## " + filename)

        rmse = Parallel(n_jobs=4)(
            delayed(thread)(j, filename, nbInputs) for j in range(10)
        )

        print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
        print("%5.2f)" % np.std(rmse))

