from abc          import ABCMeta, abstractmethod
from numpy.random import RandomState
import tensorflow as tf

class Solver(object):
    __metaclass__ = ABCMeta

    def __init__(self, id, run, nbInputs, rs=None):
        self.id       = id
        self.run      = run
        self.nbInputs = nbInputs
        self.rs       = rs if rs else RandomState(None)

    @abstractmethod
    def train(self, data, validationData, nbEpochs=100):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def solve(self, x):
        pass

    def close(self):
        pass
