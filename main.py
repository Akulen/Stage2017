from dt               import RF
from math             import sqrt
from nn               import NNF
from rnf              import RNF1, RNF2
from sklearn.ensemble import RandomForestRegressor
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
nbNeurones = 2**(maxProf+1)
nbIter     = 5

for i, (filename, nbInputs) in enumerate(datafiles[2:3]):
    print("## " + filename)
    rmse     = []
    for _ in range(3):
        #solver = NNF(nbInputs, nbNeurones, nbIter)
        solver = RF(nbInputs, nbNeurones, nbIter)
        #solver = RNF1(nbInputs, maxProf, nbNeurones, nbIter)
        #solver = RNF1(nbInputs, maxProf, nbNeurones, nbIter, sparse=False)
        #solver = RNF2(nbInputs, maxProf, nbNeurones, nbIter)
        #solver = RNF2(nbInputs, maxProf, nbNeurones, nbIter, sparse=False)

        rmse.append(evaluateSolver(solver, getData(filename, nbInputs, 1))[0])

    print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
    print("%5.2f)" % np.std(rmse))

