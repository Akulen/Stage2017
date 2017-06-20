import random
import tensorflow as tf
import utils
from solver import Solver
from forest import Forest
from math import sqrt

def randomRange(fan_in, fan_out):
    return sqrt(6 / (fan_in + fan_out))

class NN(Solver):
    def __init__(self, id, run, nbInputs, nbFeatures):
        super().__init__(id, run, nbInputs)
        self.nbFeatures = nbFeatures

        self.x = tf.placeholder(tf.float32, [None, nbInputs])

        self.summaries = []
        self.layers = []
        y = self.x
        dimensions = [nbInputs, nbFeatures-1, nbFeatures, 1]
        gamma = [100, 1]
        for i in range(3):
            init_W = tf.truncated_normal(dimensions[i:i+2], dtype=tf.float32,
                    seed=random.randint(-10**5, 10**5))
            W = tf.Variable(initial_value=init_W)
            init_b = tf.truncated_normal([dimensions[i+1]], dtype=tf.float32,
                    seed=random.randint(-10**5, 10**5))
            b = tf.Variable(initial_value=init_b)
            #r = randomRange(dimensions[i], dimensions[i+1])
            #init_W = tf.random_uniform(dimensions[i:i+2], -r, r,
            #        seed=random.randint(0, 10**9))
            #W = tf.Variable(initial_value=init_W)
            #init_b = tf.random_uniform([dimensions[i+1]], -r, r,
            #        seed=random.randint(0, 10**9))
            #b = tf.Variable(initial_value=init_b)
            if i == 2:
                y = tf.matmul(y, W) + b
            else:
                y = tf.tanh(gamma[i] * (tf.matmul(y, W) + b))
            self.layers.append((W, b, y))

        self.y = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)
        self.vloss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)
        
        self.summaries.append(tf.summary.scalar("loss/" + str(self.id), tf.sqrt(self.loss)))
        self.summaries.append(tf.summary.scalar("vloss/" + str(self.id), tf.sqrt(self.vloss)))
        #self.summaries.append(tf.summary.histogram("lossHisto", self.loss))

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()

        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train/'+self.run, self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./test/'+self.run, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

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
                summary, _ = self.sess.run([self.summaries[0], self.train_step],
                        feed_dict={self.x: batch_xs, self.y: batch_ys})
                if j == 0:
                    self.train_writer.add_summary(summary, i)
            summary, closs = self.sess.run([self.summaries[-1], self.vloss], feed_dict={self.x: vx, self.y: vy})
            self.train_writer.add_summary(summary, i)
            if i % 10 == 9:
                print("Epoch #" + str(i) + ": " + str(closs))
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
        self.nbFeatures = nbFeatures

    def train(self, data, validation, nbEpochs=100):
        self.data = data
        self.validation = validation
        self.nbEpochs = nbEpochs

    def evaluate(self, data):
        z = [[] for i in range(len(data))]
        for i in range(self.nbIter):
            nn = NN(i, self.run, self.nbInputs, self.nbFeatures)
            nn.train(self.data, self.validation, self.nbEpochs)
            for j in range(len(data)):
                x, y = data[j]
                z[j].append(nn.solve([x]))
            print(i)
        for i in range(len(data)):
            z[i] = sum(z[i]) / self.nbIter
        return utils.evaluate(z, [y[0] for _, y in data])





