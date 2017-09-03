from dn                     import DN1, DN2
from dt                     import DT, RF
from joblib                 import Parallel, delayed
from math                   import tau, sin, sqrt
from nn                     import NNF1, NNF2
from rnf                    import RNF1, RNF2
from sklearn.ensemble       import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow             import Session
from utils                  import evaluate, getData, sq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import sys
import tensorflow as tf

rs = np.random.RandomState(424242)

def noise():
    return rs.normal(0, noiseVar)

def f0(x):
    assert 0 <= x and x <= 1
    if x < 1/3:
        return 1-3*x
    if x < 2/3:
        return 3*x-1
    return 0

def f1(x):
    return (2*x - 1)**2

def f2(x):
    return 1 if x > 1/2 else -1

def f3(x):
    return f2((sin(5 * tau * x) + 1) / 2)

fs = [f0, f1, f2, f3]
assert len(sys.argv) > 1
f = fs[int(sys.argv[1])]

trainSize  = 100
maxProf    = 5
nbNeurones = 2**maxProf
nbIter     = 3
noiseVar   = 0.01

trainX = [rs.rand() for _ in range(trainSize)]; trainX.sort()
trainY = list(map(lambda x : f(x)+noise(), trainX))
index = list(range(trainSize))
rs.shuffle(index)
trainX = [trainX[i] for i in index]
trainY = [trainY[i] for i in index]

X = np.linspace(0, 1, 10**3)
Xle = [[x] for x in X]
Y = np.array([f(x) for x in X])

solvers = [
    ("Neural Net", lambda n, sess, rs :
        NNF2(n, nbNeurones, sess=sess, debug=True, nbIter=nbIter, pref="base", rs=rs)
    ),
    ("Random Forest", lambda n, sess, rs :
        RF(n, maxProf, nbIter=nbIter, pref="base", rs=rs)
    ),
    ("Random Forest - ET, Prof5", lambda n, sess, rs :
        DT(0, "", n, maxProf, None, nbIter=nbIter, treeType=ExtraTreesRegressor, rs=rs)
    ),
    ("Random Forest - ET", lambda n, sess, rs :
        DT(0, "", n, None, None, nbIter=nbIter, treeType=ExtraTreesRegressor, rs=rs)
    ),
    ("Learning Forest mean - ET", lambda n, sess, rs :
        RNF1(n, maxProf, nbNeurones, fix=[False, True, False], sess=sess, sparse=True,
            useEt=True, nbIter=nbIter, pref="learning-et", rs=rs)
    ),
    ("Neural Random Forest mean - Full", lambda n, sess, rs :
        RNF1(n, maxProf, nbNeurones, sess=sess, sparse=False, useEt=False,
            nbIter=nbIter, pref="full", rs=rs)
    ),
    ("Random Neural Forest mean - ET, Full", lambda n, sess, rs :
        RNF1(n, maxProf, nbNeurones, sess=sess, useEt=True, nbIter=nbIter,
            pref="et-full", rs=rs)
    ),
    ("Random Neural Forest - Full", lambda n, sess, rs :
        RNF2(n, maxProf, nbNeurones, sess=sess, useEt=False, nbIter=nbIter,
            pref="full", rs=rs)
    ),
    ("Random Neural Forest - ET, Full", lambda n, sess, rs :
        RNF2(n, maxProf, nbNeurones, sess=sess, useEt=True, nbIter=nbIter,
            pref="et-full", rs=rs)
    ),
    ("Decision Network mean", lambda n, sess, rs :
        DN1(n, maxProf, nbNeurones, sess=sess, useEt=False, nbIter=nbIter,
            pref="", rs=rs)
    ),
    ("Decision Network joint", lambda n, sess, rs :
        DN2(n, maxProf, nbNeurones, sess=sess, useEt=False, nbIter=nbIter,
            pref="", rs=rs)
    ),
    ("Decision Network mean - ET", lambda n, sess, rs :
        DN1(n, maxProf, nbNeurones, sess=sess, useEt=True, nbIter=nbIter,
            pref="et", rs=rs)
    ),
    ("Decision Network - ET", lambda n, sess, rs :
        DN2(n, maxProf, nbNeurones, sess=sess, useEt=True, nbIter=nbIter,
            pref="et", rs=rs)
    )
]

sess = tf.Session()

queue = []
if len(sys.argv) == 2:
    queue = solvers[1:8]
else:
    for _i in sys.argv[2:]:
        i = int(_i)
        queue.append(solvers[i])

trainData = [([trainX[i]], [trainY[i]]) for i in range(trainSize)]

seed = rs.randint(1E9)

def process(i):
    rss    = np.random.RandomState(seed)
    s      = queue[i]
    sess   = tf.Session()
    solver = s[1](1, sess, np.random.RandomState(rss.randint(1E9)))
    fn     = solver.train(trainData[:], trainData[:], nbEpochs=0, logEpochs=Xle, sampler=np.random.RandomState(rss.randint(1E9)))
    y      = solver.solve([[x] for x in X])
    print("%-40s %9.5f" % (s[0] + ":", evaluate(y, Y)))
    solver.close()
    return fn

fns = Parallel(n_jobs=4, backend="threading")(
    delayed(process)(i) for i in range(len(queue))
)

fns = np.swapaxes(np.array(fns), 0, 1)

my_dpi=96
fig, ax = plt.subplots(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid()
ax.figure.canvas.draw()
ax.plot(X, Y, lw=2)
ax.plot(trainX, trainY, "+", lw=2)
lines = [ax.plot([], [], lw=2, label=queue[i][0])[0] for i in range(len(queue))]
ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

def init():
    ax.set_xlim(-.1, 1.1)
    ax.set_ylim(min(Y)-.5, max(Y)+.5)
    return tuple(lines)

def run(data):
    for i in range(len(data)):
        lines[i].set_data(X, data[i])
    return tuple(lines)

ani = animation.FuncAnimation(fig, run, fns, interval=100,
        init_func=init)
#ani.save('train.mp4', fps=10, dpi=96)
plt.show()

