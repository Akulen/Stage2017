import numpy as np
import random
import tensorflow as tf
import utils
from joblib import Parallel, delayed
from solver import Solver
from forest import Forest
from math import sqrt

#r = randomRange(dimensions[i], dimensions[i+1])
def randomRange(fan_in, fan_out):
    return sqrt(6 / (fan_in + fan_out))

class NN(Solver):
    def __init__(self, id, run, nbInputs, layers, connectivity=None, weight=None, bias=None):
        super().__init__(id, run, nbInputs)
        self.layers = layers
        self.summaries = []
        self.layers = []

        gamma = [y for _, y in layers]
        self.dimensions = [nbInputs] + [x for x, _ in layers] + [1]

        connectivity = self.initConnectivity(layers, connectivity)
        weight       = self.initWeight(layers, weight)
        bias         = self.initBias(layers, bias)

        self.x = tf.placeholder(tf.float32, [None, nbInputs])

        y = self.x

        for i in range(len(layers)+1):
            b = tf.Variable(initial_value=bias[i])
            W = tf.Variable(initial_value=weight[i])

            if i < len(layers) - 1:
                C = tf.constant(connectivity[i], dtype=tf.float32)
                W = W * C

            y = tf.sparse_matmul(y, W, b_is_sparse=True) + b
            if i < len(layers) - 1:
                y = tf.tanh(gamma[i] * y)

            self.layers.append((W, b, y))

        self.y = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)
        self.vloss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()

        #self.summaries.append(tf.summary.scalar("loss/" + str(self.id), tf.sqrt(self.loss)))
        self.summaries.append(tf.summary.scalar("vloss/" + str(self.id), tf.sqrt(self.vloss)))
        #self.summaries.append(tf.summary.histogram("lossHisto", self.loss))
        self.summary = tf.summary.merge_all()
        #self.train_writer = tf.summary.FileWriter('./train/'+self.run, self.sess.graph)
        #self.test_writer = tf.summary.FileWriter('./test/'+self.run, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def initConnectivity(self, layers, connectivity):
        if connectivity is None:
            connectivity = [None] * len(layers)
        for i in range(len(connectivity)):
            if connectivity[i] is None:
                connectivity[i] = np.ones((self.dimensions[i], self.dimensions[i+1]))
        return connectivity

    def initWeight(self, layers, weight):
        if weight is None:
            weight = [None] * (len(layers) + 1)
        for i in range(len(weight)):
            if weight[i] is None:
                weight[i] = tf.truncated_normal(self.dimensions[i:i+2], dtype=tf.float32,
                        seed=random.randint(-10**5, 10**5))
                #init_W = tf.random_uniform(dimensions[i:i+2], -r, r,
                #        seed=random.randint(0, 10**9))
            else:
                weight[i] = tf.constant(weight[i], dtype=tf.float32)
        return weight

    def initBias(self, layers, bias):
        if bias is None:
            bias = [None] * (len(layers) + 1)
        for i in range(len(bias)):
            if bias[i] is None:
                bias[i] = tf.truncated_normal([self.dimensions[i+1]], dtype=tf.float32,
                        seed=random.randint(-10**5, 10**5))
                #init_b = tf.random_uniform([dimensions[i+1]], -r, r,
                #        seed=random.randint(0, 10**9))
            else:
                bias[i] = tf.constant(bias[i], dtype=tf.float32)
        return bias

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        #if i % 10 == 0:  # Record summaries and test-set accuracy
        #    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        #    test_writer.add_summary(summary, i)
        #    print('Accuracy at step %s: %s' % (i, acc))

        vx, vy = map(list, zip(* validation))
        loss = float("inf")
        saver = tf.train.Saver()
        for i in range(nbEpochs):
            for j in range(2*len(data) // batchSize + 1):
                batch_xs, batch_ys = utils.selectBatch(data, batchSize)
                self.sess.run([self.train_step],
                        feed_dict={self.x: batch_xs, self.y: batch_ys})
                #if j == 0:
                #    self.train_writer.add_summary(summary, i)
            summary, closs = self.sess.run([self.summaries[0], self.vloss], feed_dict={self.x: vx, self.y: vy})
            #self.train_writer.add_summary(summary, i)
            #if i % 10 == 9:
            #    print("Epoch #" + str(i) + ": " + str(closs))
            if closs < loss:
                loss = closs
                saver.save(self.sess, "/tmp/sess" + str(self.id) + ".ckpt")
        saver.restore(self.sess, "/tmp/sess" + str(self.id) + ".ckpt")


    def evaluate(self, data):
        xs, ys = map(list, zip(* data))

        #correct_prediction = tf.equal(tf.argmax(self.layers[-1][2], 1), tf.argmax(self.y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(self.sess.run(accuracy, feed_dict={self.x: xs, self.y: ys}))

        rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.layers[-1][2]))
        _, var = tf.nn.moments(tf.square(self.y - self.layers[-1][2]) / len(data), axes=[0])
        devi = tf.sqrt(var)
        results = self.sess.run([rmse, devi], feed_dict={self.x: xs, self.y: ys})
        return results[0], results[1]

    def solve(self, x):
        return self.sess.run(self.layers[-1][2], feed_dict={self.x: x})

class NNF(Forest):
    def __init__(self, nbInputs, nbFeatures, nbIter=-1):
        super().__init__(nbIter)
        self.nbInputs = nbInputs
        self.layers = [(nbFeatures-1, 100), (nbFeatures, 1)]

    def thread(self, id, data):
        nn = NN(id, self.run, self.nbInputs, self.layers)
        nn.train(self.data, self.validation, self.nbEpochs)
        z = [None] * len(data)
        for j in range(len(data)):
            x, y = data[j]
            z[j] = nn.solve([x])
        return z

    def train(self, data, validation, nbEpochs=100):
        self.data = data
        self.validation = validation
        self.nbEpochs = nbEpochs

    def evaluate(self, data):
        z = [0] * len(data)
        res = Parallel(n_jobs=16)(
            delayed(self.thread)(i, data) for i in range(self.nbIter)
        )
        for j in range(len(data)):
            for i in range(self.nbIter):
                z[j] += res[i][j]
            z[j] = sum(z[j]) / len(data)
        return utils.evaluate(z, [y[0] for _, y in data])





