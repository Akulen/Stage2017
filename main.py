from dt import RF
from math import sqrt
from nn import NNF
from rnf import RNF1, RNF2
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random

def getData(filename, nbX, nbY):
    data = []
    for line in open(filename, "r"):
        raw = list(map(float, line.split()))
        data.append((raw[nbX:nbX+nbY], raw[0:nbX]))
    return data

def evaluateSolver(solver, data):
    random.shuffle(data)
    n              = len(data)
    trainData      = data[:n//2]
    validationData = data[n//2:3*n//4]
    testData       = data[3*n//4:]

    solver.train(trainData, validationData)
    return solver.evaluate(testData)

datafiles = [("./data/autoMPG/autoMPG.data",     7 ),
             ("./data/housing/housing.data",     13),

             ("./data/forest/forest.data",       10),
             ("./data/wisconsin/wisconsin.data", 32),
             ("./data/concrete/concrete.data",   8 )]
dataset = [getData(filename, 1, nbInputs) for filename, nbInputs in datafiles]

for i, data in enumerate(dataset):
    print(datafiles[i][0])
    nbInputs = datafiles[i][1]
    rmse = []
    for _ in range(3):
        #solver = RF(nbInputs, 6, 30)
        #solver = NNF(nbInputs, 128, 30)
        #solver = RNF1(nbInputs, 6, 128, 30)
        #solver = RNF1(nbInputs, 6, 128, 3, False)
        #solver = RNF2(nbInputs, 6, 128, 30)
        solver = RNF2(nbInputs, 6, 128, 3, False)

        rmse.append(evaluateSolver(solver, data)[0])

    print("%.2f (" % (sum(rmse) / len(rmse)), end='')
    print("%.2f)" % np.std(rmse))

#x, y = map(list, zip(* data))
#y = [yy[0] for yy in y]
#rf2 = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=3)
#rf2.fit(x[:n//2], y[:n//2])
#
#z = rf2.predict(x[3*n//4:])
#
#z = np.array(z)
#y = np.array(y[3*n//4:])
#sqError = (z-y)*(z-y)
#mse = np.mean(sqError)
#X = [i for i in range(len(sqError))]
#Y0 = [x for x in sqError]
#Y1 = sqError
#Y1.sort()
#Y2 = [mse] * len(sqError)
#rmse = sqrt(mse)
#devi = np.std(sqError)
##my = sum(y) / len(y)
##resu = [square(z[i] - y[i]) for i in range(len(y))]
##rmse = sqrt(sum(resu) / len(y))
##resu = [square(z[i] - my) for i in range(len(y))]
##devi = sqrt(sum(resu) / len(y))
#
#print("RMSE: %.4f" % rmse)
#print("devi: %.4f" % devi)
#
