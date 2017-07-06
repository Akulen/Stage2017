from math import sqrt
import datetime
import numpy as np
import random
import re

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def dt2nn(dt, tree, a, b, c, n):
    connectivity = [
        np.zeros((a, b)),
        np.zeros((b, c))
    ]
    weight = [
        np.zeros((a, b)),
        np.zeros((b, c)),
        np.zeros((c, 1))
    ]
    bias = [
        np.zeros(b),
        np.zeros(c),
        np.zeros(1)
    ]

    nbNodes      = tree.node_count
    father, side = makeTree(tree)
    nodes, leafs = indexNodes(tree), indexLeafs(tree)
    nodeMap      = revIndex(nodes)

    for j, node in enumerate(nodes):
        connectivity[0][tree.feature[node]][j] = 1.
        weight[0][tree.feature[node]][j]       = 1.
        bias[0][j]                             = - tree.threshold[node]
    
    for j, leaf in enumerate(leafs):
        l   = 0
        v   = side[leaf]
        cur = father[leaf]
        while cur != -1:
            curI                     = nodeMap[cur]
            connectivity[1][curI][j] = 1.
            weight[1][curI][j]       = v

            l   += 1
            v    = side[cur]
            cur  = father[cur]

        bias[1][j] = 0.5 - l

        weight[2][j][0]  = tree.value[leaf] / (2 * n)
        bias[2][0]      += weight[2][j][0]

    return connectivity, weight, bias

def evaluate(z, y):
    z, y = np.array(z), np.array(y)
    return rmse(y.flatten(), z.flatten())

def getData(filename, nbX, nbY):
    data = []
    for line in open("./data/" + filename + "/" + filename + ".data", "r"):
        raw = list(map(float, line.split()))
        data.append((raw[nbY:nbY+nbX], raw[0:nbY]))
    return data

def indexNodes(tree):
    nodes = []
    for i in range(tree.node_count):
        if tree.children_left[i] >= 0:
            nodes.append(i)
    return nodes

def indexLeafs(tree):
    leafs = []
    for i in range(tree.node_count):
        if tree.children_left[i] < 0:
            leafs.append(i)
    return leafs

def makeTree(tree):
    father = [-1] * tree.node_count
    side   = [0] * tree.node_count
    for i in range(tree.node_count):
        if tree.children_left[i] >= 0:
            father[tree.children_left[i]]  = i
            father[tree.children_right[i]] = i
            side[tree.children_left[i]]    = -1.
            side[tree.children_right[i]]   = 1.
    return father, side

def revIndex(index):
    mp = {}
    for i in range(len(index)):
        mp[index[i]] = i
    return mp

def rmse(a, b):
    return sqrt(np.mean(sq(a-b)))

def selectBatch(data, batchSize, replace=True, unzip=True):
    ret = []
    for i in range(batchSize):
        j = random.randint(0 if replace else i, len(data)-1)
        ret.append(data[j])
        if not replace:
            data[i], data[j] = data[j], data[i]
    return map(list, zip(* ret)) if unzip else ret

def sq(x):
    return x*x

def zipData(data):
    x, y = map(list, zip(* data))
    return x, [yy[0] for yy in y]

