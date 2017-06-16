import tensorflow as tf

class Solver(object):
    def __init__(self, id, run, nbInputs):
        self.id       = id
        self.run      = run
        self.nbInputs = nbInputs

    def train(self, data, validationData, nbEpochs=100, batchSize=32):
        pass
    
    def evaluate(self, data):
        pass

    def solve(self, x):
        pass
