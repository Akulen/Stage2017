from forest import ParallelForest
from joblib import Parallel, delayed
from math   import sqrt
from solver import Solver
import numpy as np
import random
import tensorflow as tf
import utils

#r = randomRange(dimensions[i], dimensions[i+1])
def randomRange(fan_in, fan_out):
    return sqrt(6 / (fan_in + fan_out))

class NN(Solver):
    def __init__(self, id, run, nbInputs, layers, connectivity=None,
            weight=None, bias=None, sess=None, pref="", debug=False,
            use_relu=False):
        super().__init__(id, run, nbInputs)
        self.pref      = pref
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
                W = tf.Variable(initial_value=weight[i][j])

                if i < len(layers) - 1:
                    C = tf.constant(connectivity[i][j], dtype=tf.float32)
                    W = W * C

                y_id = j
                if len(y) == 1:
                    y_id = 0
                ny[j] = tf.sparse_matmul(y[y_id], W, b_is_sparse=True) + b

                if i < len(layers):
                    fAct = tf.nn.relu if use_relu else tf.tanh
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
            self.summaries.append(tf.summary.scalar("vloss/" + str(self.id), self.vloss))
        self.summary = tf.summary.merge_all()
        if self.debug:
            self.train_writer = tf.summary.FileWriter('./train/'+self.run, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def initConnectivity(self, layers, connectivity):
        if connectivity is None:
            connectivity = [None] * len(layers)
        for i in range(len(connectivity)):
            if connectivity[i] is None:
                connectivity[i] = [np.ones((self.dimensions[i], self.dimensions[i+1]))]
        return connectivity

    def initWeight(self, layers, weight):
        if weight is None:
            weight = [None] * (len(layers) + 1)
        for i in range(len(weight)):
            if weight[i] is None:
                weight[i] = [tf.truncated_normal(self.dimensions[i:i+2], dtype=tf.float32,
                        seed=random.randint(-10**5, 10**5))]
                #init_W = tf.random_uniform(dimensions[i:i+2], -r, r,
                #        seed=random.randint(0, 10**9))
            else:
                for j in range(len(weight[i])):
                    weight[i][j] = tf.constant(weight[i][j], dtype=tf.float32)
        return weight

    def initBias(self, layers, bias):
        if bias is None:
            bias = [None] * (len(layers) + 1)
        for i in range(len(bias)):
            if bias[i] is None:
                bias[i] = [tf.truncated_normal([self.dimensions[i+1]], dtype=tf.float32,
                        seed=random.randint(-10**5, 10**5))]
                #init_b = tf.random_uniform([dimensions[i+1]], -r, r,
                #        seed=random.randint(0, 10**9))
            else:
                for j in range(len(bias[i])):
                    bias[i][j] = tf.constant(bias[i][j], dtype=tf.float32)
        return bias

    def train(self, data, validation, nbEpochs=100, batchSize=32, logEpochs=False):
        loss = float("inf")
        saver = tf.train.Saver()
        if logEpochs:
            testData = [[x] for x in np.linspace(0, 1, 10**3)]
            fns = [None] * nbEpochs
        for i in range(nbEpochs):
            for j in range(len(data) // batchSize + 1):
                batch_xs, batch_ys = utils.selectBatch(data, batchSize)
                self.sess.run([self.train_step],
                        feed_dict={self.x: batch_xs, self.y: batch_ys})
            if self.debug:
                summary, closs = self.evaluate(validation)
                self.train_writer.add_summary(summary, i)
                if i % 10 == 9:
                    print("Epoch #" + str(i) + ": " + str(closs))
            else:
                closs = self.evaluate(validation)
            if closs < loss:
                loss = closs
                saver.save(self.sess, "/tmp/sess" + self.pref + "-" + str(self.id) + ".ckpt")
                if logEpochs:
                    y = self.solve(testData)
                    fns[i] = [_y[0] for _y in y]
            elif logEpochs:
                fns[i] = fns[i-1][:]
        saver.restore(self.sess, "/tmp/sess" + self.pref + "-" + str(self.id) + ".ckpt")
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
        super().close()
        self.sess.close()



class NNF(ParallelForest):
    def __init__(self, nbInputs, nbFeatures, nbIter=-1, nbJobs=8, sess=None,
            use_relu=False, pref="", debug=False):
        self.nbInputs = nbInputs
        self.layers   = [(nbFeatures, 100), (nbFeatures, 1)]
        self.use_relu = use_relu
        self.sess     = sess
        self.debug    = debug

        super().__init__(nbIter, nbJobs, "neural-net-" + pref)

    def createSolver(self, id):
        return NN(id, self.run, self.nbInputs, self.layers, sess=self.sess,
                use_relu=self.use_relu, pref=self.pref, debug=self.debug)





