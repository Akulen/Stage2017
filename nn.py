import tensorflow as tf
import random
import datetime
import re
from solver import Solver

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def selectBatch(data, batchSize):
    for i in range(batchSize):
        j = random.randint(i, len(data)-1)
        data[i], data[j] = data[j], data[i]
    return map(list, zip(* data[:batchSize]))

class NN(Solver):
    def __init__(self, nbInputs, nbFeatures):
        super().__init__(nbInputs)
        self.nbFeatures = nbFeatures

        self.x = tf.placeholder(tf.float32, [None, nbInputs])

        self.summaries = []
        self.layers = []
        y = self.x
        dimensions = [nbInputs, nbFeatures-1, nbFeatures, 1]
        for i in range(3):
            init_W = tf.truncated_normal(dimensions[i:i+2], dtype=tf.float32,
                    seed=random.randint(0, 10**9))
            W = tf.Variable(initial_value=init_W)
            init_b = tf.truncated_normal([dimensions[i+1]], dtype=tf.float32,
                    seed=random.randint(0, 10**9))
            b = tf.Variable(initial_value=init_b)
            if i == 2:
                y = tf.matmul(y, W) + b
            else:
                y = tf.tanh(tf.matmul(y, W) + b)
            self.layers.append((W, b, y))

        self.y = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)
        self.vloss = tf.losses.mean_squared_error(self.layers[-1][2], self.y)
        
        self.summaries.append(tf.summary.scalar("loss", tf.sqrt(self.loss)))
        self.summaries.append(tf.summary.scalar("vloss", tf.sqrt(self.vloss)))
        #self.summaries.append(tf.summary.histogram("lossHisto", self.loss))

        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.InteractiveSession()

        self.summary = tf.summary.merge_all()
        self.time = custom_iso()
        self.train_writer = tf.summary.FileWriter('./train/'+self.time, self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./test', self.sess.graph)

        tf.global_variables_initializer().run()

    def train(self, data, validation, nbEpochs=100, batchSize=32):
        #if i % 10 == 0:  # Record summaries and test-set accuracy
        #    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        #    test_writer.add_summary(summary, i)
        #    print('Accuracy at step %s: %s' % (i, acc))

        vx, vy = map(list, zip(* validation))
        loss = float("inf")
        saver = tf.train.Saver()
        for i in range(nbEpochs):
            for j in range(len(data) // batchSize + 1):
                batch_xs, batch_ys = selectBatch(data, batchSize)
                summary, _ = self.sess.run([self.summaries[0], self.train_step],
                        feed_dict={self.x: batch_xs, self.y: batch_ys})
                if j == 0:
                    self.train_writer.add_summary(summary, i)
            summary, closs = self.sess.run([self.summaries[-1], self.vloss], feed_dict={self.x: vx, self.y: vy})
            self.train_writer.add_summary(summary, i)
            print("Epoch:" + str(i) + " -> " + str(closs))
            if closs < loss:
                loss = closs
                saver.save(self.sess, "/tmp/sess.ckpt")
        saver.restore(self.sess, "/tmp/sess.ckpt")


    def evaluate(self, data):
        xs, ys = map(list, zip(* data))

        #correct_prediction = tf.equal(tf.argmax(self.layers[-1][2], 1), tf.argmax(self.y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(self.sess.run(accuracy, feed_dict={self.x: xs, self.y: ys}))

        rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.layers[-1][2]))
        _, var = tf.nn.moments(tf.square(self.y - self.layers[-1][2]) / len(data), axes=[0])
        devi = tf.sqrt(var)
        results = self.sess.run([rmse, devi], feed_dict={self.x: xs, self.y: ys})
        print("RMSE: %.4f" % results[0])
        print("devi: %.4f" % results[1])





