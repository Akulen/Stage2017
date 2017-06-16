from nn import NN
import random

def getData(filename, nbX, nbY):
    data = []
    for line in open(filename, "r"):
        raw = list(map(float, line.split()))
        data.append((raw[nbX:nbX+nbY], raw[0:nbX]))
    return data

data           = getData("./data/autoMPG/autoMPG.data", 1, 7)
random.shuffle(data)
n              = len(data)
trainData      = data[:n//2]
validationData = data[n//2:3*n//4]
testData       = data[3*n//4:]

net = NN(7, 64)
net.train(trainData, validationData)
net.evaluate(testData)
