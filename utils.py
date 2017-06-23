from math import sqrt
import datetime
import numpy as np
import random
import re

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def evaluate(z, y):
    z       = np.array(z)
    y       = np.array(y)
    sqError = sq(z-y)
    rmse    = sqrt(np.mean(sqError))
    devi    = np.std(sqError)
    return rmse, devi

def getData(filename, nbX, nbY):
    data = []
    for line in open("./data/" + filename + "/" + filename + ".data", "r"):
        raw = list(map(float, line.split()))
        data.append((raw[nbY:nbY+nbX], raw[0:nbY]))
    return data

def selectBatch(data, batchSize, replace=True, unzip=True):
    ret = []
    for i in range(batchSize):
        j = random.randint(0 if replace else i, len(data)-1)
        ret.append(data[j])
        if not replace:
            data[i], data[j] = data[j], data[i]
    return map(list, zip(* data[:batchSize])) if unzip else data[:batchSize]

def sq(x):
    return x*x

