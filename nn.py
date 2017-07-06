from forest import Forest, ParallelForest
from joblib import Parallel, delayed
from math   import sqrt
from solver import Solver
import numpy as np
import os
import random
import tensorflow as tf
import utils

#r = randomRange(dimensions[i], dimensions[i+1])
def randomRange(fan_in, fan_out):
    return sqrt(6 / (fan_in + fan_out))

class NN(Solver):
    def __init__(self, id, run, nbInputs, layers, connectivity=None,
            weight=None, bias=None, useRelu=False, fixMiddle=False, sess=None,
            debug=False):
        super().__init__(id, run, nbInputs)
        self.layers    = layers
        self.summaries = []
        self.layers    = []
        self.debug     = debug

        gamma           = [y for _, y in layers]
        self.dimensions = [nbInputs] + [x for x, _ in layers] + [1]

        connectivity = self.initConnectivity(layers, connectivity)
        weight       = self.initWeight(layers, weight)
        bias         = self.initBias(layers, bias)

        self.x = tf.placeholder(tf.float32, [None, nbInputs], name="Input")

        y = [self.x]
        ny = []

        for i in range(len(layers)+1):
            if i < len(layers):
                assert len(connectivity[i]) == len(weight[i])
            assert len(bias[i]) == len(weight[i])

            layer = []
            ny = [None] * len(bias[i])
            for j in range(len(bias[i])):
                b = tf.Variable(initial_value=bias[i][j])
                if j == 1 and fixMiddle:
                    W = weight[i][j]
                else:
                    W = tf.Variable(initial_value=weight[i][j])

                if i < len(layers) - 1:
                    C = tf.constant(connectivity[i][j], dtype=tf.float32)
                    W = W * C

                y_id = j
                if len(y) == 1:
                    y_id = 0
                ny[j] = tf.sparse_matmul(y[y_id], W, b_is_sparse=True) + b

                if i < len(layers):
                    fAct = tf.nn.relu if useRelu else tf.tanh
                    ny[j] = fAct(gamma[i] * ny[j])

                layer.append((W, b, ny[j]))

            if i == len(layers):
                ny = [tf.add_n(ny)]

            y = ny

            self.layers.append(layer)

        self.output = y[0]
        self.y = tf.placeholder(tf.float32, [None, 1])

        self.loss  = tf.losses.mean_squared_error(self.output, self.y)
        self.vloss = tf.sqrt(tf.losses.mean_squared_error(self.output, self.y))

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session() if sess is None else sess

        if self.debug:
            self.summaries.append(tf.summary.scalar("vloss/" + self.id, self.vloss))
        self.summary = tf.summary.merge_all()
        if self.debug:
            self.train_writer = tf.summary.FileWriter('./train/'+self.run, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def initConnectivity(self, layers, connectivity):
        if connectivity is None:
            connectivity = [None] * len(layers)
        for i in range(len(connectivity)):
            if connectivity[i] is None:
                connectivity[i] = [None]
            for j in range(len(connectivity[i])):
                if connectivity[i][j] is None:
                    connectivity[i][j] = np.ones((self.dimensions[i],
                        self.dimensions[i+1]))
        return connectivity

    def initWeight(self, layers, weight):
        if weight is None:
            weight = [None] * (len(layers) + 1)
        for i in range(len(weight)):
            if weight[i] is None:
                weight[i] = [None]
                #init_W = tf.random_uniform(dimensions[i:i+2], -r, r,
                #        seed=random.randint(0, 10**9))
            for j in range(len(weight[i])):
                if weight[i][j] is None:
                    weight[i][j] = tf.truncated_normal(self.dimensions[i:i+2],
                            dtype=tf.float32, seed=random.randint(-10**5, 10**5))
                else:
                    weight[i][j] = tf.constant(weight[i][j], dtype=tf.float32)
        return weight

    def initBias(self, layers, bias):
        if bias is None:
            bias = [None] * (len(layers) + 1)
        for i in range(len(bias)):
            if bias[i] is None:
                bias[i] = [None]
                #init_b = tf.random_uniform([dimensions[i+1]], -r, r,
                #        seed=random.randint(0, 10**9))
            for j in range(len(bias[i])):
                if bias[i][j] is None:
                    bias[i][j] = tf.truncated_normal([self.dimensions[i+1]],
                            dtype=tf.float32, seed=random.randint(-10**5, 10**5))
                else:
                    bias[i][j] = tf.constant(bias[i][j], dtype=tf.float32)
        return bias

    def train(self, data, validation, nbEpochs=100, batchSize=32, logEpochs=False):
        loss = float("inf")
        saver = tf.train.Saver()
        if logEpochs:
            testData = [[x] for x in np.linspace(0, 1, 10**3)]
            fns = [None] * (nbEpochs + 1)
            y = self.solve(testData)
            fns[0] = [_y[0] for _y in y]
        for i in range(nbEpochs):
            for j in range(len(data) // batchSize + 1):
                batch_xs, batch_ys = utils.selectBatch(data, batchSize)
                self.sess.run([self.train_step],
                        feed_dict={self.x: batch_xs, self.y: batch_ys})
            if self.debug:
                summary, closs = self.evaluate(validation)
                self.train_writer.add_summary(summary, i)
                if i % 10 == 9:
                    print("%s: Epoch #%02d -> %9.4f" % (self.id, i, closs))
            else:
                closs = self.evaluate(validation)
            if closs < loss:
                loss = closs
                saver.save(self.sess, "/tmp/sess" + self.id + ".ckpt")
                if logEpochs:
                    y = self.solve(testData)
                    fns[i+1] = [_y[0] for _y in y]
            elif logEpochs:
                fns[i+1] = fns[i-1][:]
        saver.restore(self.sess, "/tmp/sess" + self.id + ".ckpt")
        os.remove("/tmp/sess" + self.id + ".ckpt.meta")
        os.remove("/tmp/sess" + self.id + ".ckpt.index")
        os.remove("/tmp/sess" + self.id + ".ckpt.data-00000-of-00001")
        if logEpochs:
            return fns

    def evaluate(self, data):
        xs, ys = map(list, zip(* data))
        values = self.vloss if not self.debug else [self.summaries[0], self.vloss]

        results = self.sess.run(values, feed_dict={self.x: xs, self.y: ys})
        return results

    def solve(self, x):
        return self.sess.run(self.output, feed_dict={self.x: x})

    def close(self):
        self.sess.close()
        super().close()



class NNF1(ParallelForest):
    def __init__(self, nbInputs, layerSize, useRelu=False, fixMiddle=False,
            sess=None, debug=False, nbJobs=8, nbIter=-1, pref=""):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref)
        self.pref = "neural-net-" + self.pref

        self.nbInputs  = nbInputs
        self.layers    = [(layerSize-1, 100), (layerSize, 1)]
        self.useRelu   = useRelu
        self.fixMiddle = fixMiddle
        self.sess      = sess
        self.debug     = debug

        self.initSolvers()

    def createSolver(self, id):
        return NN(self.pref + "-" + str(id), self.run, self.nbInputs,
                self.layers, useRelu=self.useRelu, fixMiddle=self.fixMiddle,
                sess=self.sess, debug=self.debug)



class NNF2(Forest):
    def __init__(self, nbInputs, layerSize, useRelu=False, fixMiddle=False,
            sess=None, debug=False, nbIter=-1, pref=""):
        super().__init__(nbIter=1, pref=pref)
        self.pref = "neural-nel-" + self.pref

        self.nbInputs  = nbInputs
        self.layers    = [(layerSize-1, 100), (layerSize, 1)]
        self.useRelu   = useRelu
        self.fixMiddle = fixMiddle
        self.sess      = sess
        self.debug     = debug

        self.connectivity = [[None] * nbIter for _ in range(2)]
        self.weight       = [[None] * nbIter for _ in range(3)]
        self.bias         = [[None] * nbIter for _ in range(3)]

        self.initSolvers()

    def createSolver(self, id):
        return NN(self.pref + "-" + str(id), self.run, self.nbInputs,
                self.layers, connectivity=self.connectivity, weight=self.weight,
                bias=self.bias, useRelu=self.useRelu, fixMiddle=self.fixMiddle,
                sess=self.sess, debug=self.debug)

    def evaluate(self, data):
        return self.iters[0].evaluate(data)

    def solve(self, x):
        return self.iters[0].solve(x)





