from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Solver(object):
    __metaclass__ = ABCMeta

    def __init__(self, id, run, nbInputs):
        self.id       = id
        self.run      = run
        self.nbInputs = nbInputs

    @abstractmethod
    def train(self, data, validationData, nbEpochs=100, batchSize=32):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def solve(self, x):
        pass
