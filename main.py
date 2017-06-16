from nn import NNF
import random

def getData(filename, nbX, nbY):
    data = []
    for line in open(filename, "r"):
        raw = list(map(float, line.split()))
        data.append((raw[nbX:nbX+nbY], raw[0:nbX]))
    return data

data           = getData("./data/concrete/concrete.data", 1, 8)
random.shuffle(data)
n              = len(data)
trainData      = data[:n//2]
validationData = data[n//2:3*n//4]
testData       = data[3*n//4:]

net = NNF(8, 64, 10)
net.train(trainData, validationData)
rmse, devi = net.evaluate(testData)
print("RMSE: %.4f" % rmse)
print("devi: %.4f" % devi)
