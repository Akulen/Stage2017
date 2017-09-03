from forest       import Forest,   ParallelForest
from joblib       import Parallel, delayed
from math         import sqrt
from numpy.random import RandomState
from solver       import Solver
import numpy as np
import tensorflow as tf
import utils

#r = randomRange(dimensions[i], dimensions[i+1])
def randomRange(fan_in, fan_out):
    return sqrt(6 / (fan_in + fan_out))

class NN(Solver):
    def __init__(self, id, run, nbInputs, layers, connectivity=None,
            weight=None, bias=None, activation=None, fix=False,
            positiveWeight=False, sess=None, debug=False, smoothW=None,
            rs=None):
        super().__init__(id, run, nbInputs, rs=rs)
        self.summaries = []
        self.layers    = []
        self.debug     = debug

        layers          = self.initLayers(layers, bias)

        gamma           = [[y for _, y in l] for l in layers]
        self.dimensions = [[nbInputs]] + [[x for x, _ in l] for l in layers] + [[1]]

        fix             = self.initFix(len(layers), fix)
        self.activation = self.initActivation(len(layers), activation)
        connectivity    = self.initConnectivity(len(layers), connectivity)
        weight          = self.initWeight(len(layers), weight)
        bias            = self.initBias(len(layers), bias)

        self.vars = []

        self.sess = tf.Session() if sess is None else sess

        tf.set_random_seed(self.rs.randint(1E9))

        with tf.name_scope(self.id):
            self.x = tf.placeholder(tf.float32, [None, nbInputs], name="Input")

            y = [self.x]
            ny = []

            self.newW = None
            if smoothW is not None:
                self.newW = []
                self.oldW = smoothW

            for i in range(len(layers)+1):
                if i < len(layers):
                    assert len(connectivity[i]) == len(weight[i])
                assert len(bias[i]) == len(weight[i])

                layer = []
                ny = [None] * len(bias[i])
                for j in range(len(ny)):
                    if fix[i][0]:
                        if i == 2 and smoothW is not None:
                            W = tf.Variable(initial_value=self.oldW[j],
                                    trainable=False)
                            self.newW.append(self.sess.run(weight[i][j]))
                            self.vars.append([W, None])
                        else:
                            W = weight[i][j]
                    else:
                        W = tf.Variable(initial_value=weight[i][j])
                        self.vars.append([W, None])
                    if i == 2 and positiveWeight:
                        W = tf.nn.softplus(W)

                    if i < len(layers):
                        C = tf.constant(connectivity[i][j], dtype=tf.float32)
                        W = W * C

                    if fix[i][1]:
                        b = bias[i][j]
                    else:
                        b = tf.Variable(initial_value=bias[i][j])
                        self.vars.append([b, None])

                    y_id = j
                    if len(y) == 1:
                        y_id = 0
                    ny[j] = tf.matmul(y[y_id], W) + b

                    if i < len(layers):
                        ny[j] = self.activation[i](gamma[i][j] * ny[j])

                    layer.append((W, b, ny[j]))
                
                if i == len(layers):
                    ny = [tf.add_n(ny)]

                y = ny

                self.layers.append(layer)

            self.output = y[0]
            self.y = tf.placeholder(tf.float32, [None, 1])

            self.loss  = tf.losses.mean_squared_error(self.output, self.y)
            self.vloss = tf.sqrt(tf.losses.mean_squared_error(self.output, self.y))

            self.train_step = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

            if self.debug:
                self.summaries.append(tf.summary.scalar("vloss/" + self.id, self.vloss))
            self.summary = tf.summary.merge_all()
            if self.debug:
                self.train_writer = tf.summary.FileWriter('./train/'+self.run, self.sess.graph)

        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=id)))

    def initFix(self, nLayer, fix):
        if fix == True or fix == False:
            fix = [fix] * (nLayer + 1)
        for i in range(len(fix)):
            if fix[i] == True or fix[i] == False:
                fix[i] = [fix[i], fix[i]]
        return fix

    def initActivation(self, nLayer, activation):
        if activation is None:
            activation = [None] * nLayer
        for i in range(len(activation)):
            if activation[i] is None:
                activation[i] = tf.tanh
        return activation

    def initConnectivity(self, nLayer, connectivity):
        if connectivity is None:
            connectivity = [None] * nLayer
        for i in range(len(connectivity)):
            if connectivity[i] is None:
                connectivity[i] = [None]
            for j in range(len(connectivity[i])):
                if connectivity[i][j] is None:
                    a = 0 if len(self.dimensions[i]) == 1 else j
                    connectivity[i][j] = np.ones((self.dimensions[i][a],
                        self.dimensions[i+1][j]))
        return connectivity

    def initWeight(self, nLayer, weight):
        if weight is None:
            weight = [None] * (nLayer + 1)
        for i in range(len(weight)):
            if weight[i] is None:
                weight[i] = [None]
                #init_W = tf.random_uniform(dimensions[i:i+2], -r, r,
                #        seed=self.rs.randint(1E9))
            for j in range(len(weight[i])):
                if weight[i][j] is None:
                    a = 0 if len(self.dimensions[i]) == 1 else j
                    b = 0 if len(self.dimensions[i+1]) == 1 else j
                    weight[i][j] = tf.truncated_normal((self.dimensions[i][a],
                        self.dimensions[i+1][b]), dtype=tf.float32,
                        seed=self.rs.randint(1E9))
                else:
                    weight[i][j] = tf.constant(weight[i][j], dtype=tf.float32)
        return weight

    def initBias(self, nLayer, bias):
        if bias is None:
            bias = [None] * (nLayer + 1)
        for i in range(len(bias)):
            if bias[i] is None:
                bias[i] = [None]
                #init_b = tf.random_uniform([dimensions[i+1]], -r, r,
                #        seed=self.rs.randint(1E9))
            for j in range(len(bias[i])):
                if bias[i][j] is None:
                    b = 0 if len(self.dimensions[i+1]) == 1 else j
                    bias[i][j] = tf.truncated_normal([self.dimensions[i+1][b]],
                            dtype=tf.float32, seed=self.rs.randint(1E9))
                else:
                    bias[i][j] = tf.constant(bias[i][j], dtype=tf.float32)
        return bias

    def initLayers(self, layers, bias):
        for i in range(len(layers)):
            if not isinstance(layers[i], list):
                layers[i] = [layers[i] for _ in range(len(bias[i]))]
        return layers

    def train(self, data, validation, nbEpochs=100, batchSize=32, logEpochs=None,
            sampler=None):
        loss = float("inf")
        for j in range(len(self.vars)):
            self.vars[j][1] = self.sess.run(self.vars[j][0])

        if logEpochs:
            y = self.solve(logEpochs)
            fns = [None] * (nbEpochs + 1)
            fns[0] = [_y[0] for _y in y]

        #print(self._evaluate(validation)[1])
        for i in range(nbEpochs):
            # Update weight if smooth transition was set
            p = min(1, 2 * (i+1) / nbEpochs)
            if self.newW is not None:
                for k in range(len(self.newW)):
                    self.sess.run(self.layers[2][k][0],
                            feed_dict={self.layers[2][k][0]: p * self.newW[k]
                            + (1 - p) * self.oldW[k]})

            # Generate all batches and launch training on them
            for j in range(len(data) // batchSize + 1):
                batch_xs, batch_ys = utils.selectBatch(data, batchSize,
                        rs=sampler)
                #if j == 0 and i % 10 == 0:
                #    print(batch_ys[0])
                #    print(self.rs.randint(1E9))
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs,
                    self.y: batch_ys})

            # Compute validation loss
            summary, curLoss = self._evaluate(validation, debug=self.debug)

            if logEpochs:
                if curLoss < loss:
                    y = self.solve(logEpochs)
                    fns[i+1] = [_y[0] for _y in y]
                else:
                    fns[i+1] = fns[i]

            # Save parameters if validation loss was improved
            if curLoss < loss:
                loss = curLoss
                for j in range(len(self.vars)):
                    self.vars[j][1] = self.sess.run(self.vars[j][0])

            if self.debug:
                self.train_writer.add_summary(summary, i)
                if i % 10 == 9:
                    print("%s: Epoch #%02d -> %9.4f" % (self.id, i, curLoss))

        # Restore best parameters
        for var, val in self.vars:
            self.sess.run(var, feed_dict={var: val})
        #print(self._evaluate(validation)[1])

        if logEpochs:
            return fns

    def _evaluate(self, data, debug=False):
        xs, ys = map(list, zip(* data))
        values = self.vloss if not debug else [self.summaries[0], self.vloss]
        
        results = self.sess.run(values, feed_dict={self.x: xs, self.y: ys})
        return results if debug else [None, results]

    def evaluate(self, data, debug=False):
        results = self._evaluate(data, debug=debug)
        return results if debug else results[1]

    def getWeights(self, layer, iter):
        return self.sess.run(self.layers[layer][iter][0])

    def getBias(self, layer, iter):
        return self.sess.run(self.layers[layer][iter][1])

    def solve(self, x):
        return self.sess.run(self.output, feed_dict={self.x: x})

    def close(self):
        self.sess.close()
        del self.sess
        del self.loss
        del self.output
        del self.summaries
        del self.summary
        del self.train_step
        if self.debug:
            del self.train_writer
        del self.vloss
        del self.x
        del self.y
        del self.layers
        super().close()



class NNF1(ParallelForest):
    def __init__(self, nbInputs, layerSize, activation=None, fix=False,
            positiveWeight=False, sess=None, debug=False, nbJobs=8, nbIter=-1,
            pref="", rs=None):
        super().__init__(nbIter=nbIter, nbJobs=nbJobs, pref=pref, rs=rs)
        self.pref = "neural-net-" + self.pref

        self.nbInputs       = nbInputs
        self.layers         = [[(layerSize-1, 100)] * nbIter,
                [(layerSize, 1)] * nbIter]
        self.activation     = activation
        self.fix            = fix
        self.positiveWeight = positiveWeight
        self.sess           = sess
        self.debug          = debug

        self.initSolvers()

    def createSolver(self, id):
        return NN(self.pref + "-" + str(id), self.run, self.nbInputs,
                self.layers, activation=self.activation,
                fix=self.fix, positiveWeight=self.positiveWeight,
                sess=self.sess, debug=self.debug, rs=self.rs)



class NNF2(Forest):
    def __init__(self, nbInputs, layerSize, activation=None, fix=False,
            positiveWeight=False, sess=None, debug=False, nbIter=-1, pref="", rs=None):
        super().__init__(nbIter=1, pref=pref, rs=rs)
        self.pref = "neural-net-" + self.pref

        self.nbInputs       = nbInputs
        self.layers         = [[(layerSize-1, 100)] * nbIter,
                [(layerSize, 1)] * nbIter]
        self.activation     = activation
        self.fix            = fix
        self.positiveWeight = positiveWeight
        self.sess           = sess
        self.debug          = debug

        self.connectivity = [[None] * nbIter for _ in range(2)]
        self.weight       = [[None] * nbIter for _ in range(3)]
        self.bias         = [[None] * nbIter for _ in range(3)]

        self.initSolvers()

    def createSolver(self, id):
        return NN(id, self.run, self.nbInputs,
                self.layers, connectivity=self.connectivity, weight=self.weight,
                bias=self.bias, activation=self.activation,
                fix=self.fix, positiveWeight=self.positiveWeight,
                sess=self.sess, debug=self.debug, rs=self.rs)

    def evaluate(self, data):
        return self.iters[0].evaluate(data)

    def solve(self, x):
        return self.iters[0].solve(x)





