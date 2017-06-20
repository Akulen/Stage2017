import datetime
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from sklearn import metrics
from math import sqrt

def sq(x):
    return x*x

def square(x):
    return x*x

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def selectBatch(data, batchSize, replace=True, unzip=True):
    ret = []
    for i in range(batchSize):
        if replace:
            j = random.randint(0, len(data)-1)
        else:
            j = random.randint(i, len(data)-1)
        ret.append(data[j])
        data[i], data[j] = data[j], data[i]
    if unzip:
        return map(list, zip(* data[:batchSize]))
    return data[:batchSize]

def evaluate(z, y):
    z = np.array(z)
    y = np.array(y)
    sqError = sq(z-y)
    mse = np.mean(sqError)
    X = [i for i in range(len(sqError))]
    Y0 = [x for x in sqError]
    Y1 = sqError
    Y1.sort()
    Y2 = [mse] * len(sqError)
    #plt.plot(X, Y0)
    #plt.plot(X, Y1)
    #plt.plot(X, Y2)
    #plt.show()
    rmse = sqrt(mse)
    devi = np.std(sqError)
    #my = sum(y) / len(y)
    #resu = [square(z[i] - y[i]) for i in range(len(y))]
    #rmse = sqrt(sum(resu) / len(y))
    #resu = [square(z[i] - my) for i in range(len(y))]
    #devi = sqrt(sum(resu) / len(y))
    return rmse, devi

