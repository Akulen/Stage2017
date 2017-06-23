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
dataset = [getData(filename, 1, nbInputs) for filename, nbInputs in datafiles]

for i, data in enumerate(dataset):
    print(datafiles[i][0])
    nbInputs = datafiles[i][1]
    rmse     = []
    for _ in range(3):
        #solver = RF(nbInputs, 6, 30)
        #solver = NNF(nbInputs, 128, 30)
        #solver = RNF1(nbInputs, 6, 128, 30)
        #solver = RNF1(nbInputs, 6, 128, 3, False)
        #solver = RNF2(nbInputs, 6, 128, 30)
        solver = RNF1(nbInputs, 6, 128, 3, False)

        rmse.append(evaluateSolver(solver, data)[0])

    print("%5.2f (" % (sum(rmse) / len(rmse)), end='')
    print("%5.2f)" % np.std(rmse))

