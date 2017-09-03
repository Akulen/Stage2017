from math         import e, log, sqrt
from numpy.random import RandomState
import datetime
import numpy as np
import re
import resource

def buildTree(p, iN, iL, balance=False):
    if len(iN) == 1:
        assert len(iL) == 2
        left = 0 if p[0][0] + p[1][1] > p[0][1] + p[1][0] else 1
        return [iN[0], [iL[left]], [iL[1-left]]]

    r = getRoot(p, balance)

    LV = []
    for i in range(len(iL)):
        LV.append((p[r][i] - p[r+len(p)//2][i], i))
    LV.sort()
    leftL = [LV[-1][1]]; rightL = [LV[0][1]]
    for i in range(1, len(iL)-1):
        if LV[i][0] < 0:
            rightL.append(LV[i][1])
        else:
            leftL.append(LV[i][1])
    assert len(leftL) > 0
    assert len(rightL) > 0
    ilL = [iL[i] for i in leftL]
    irL = [iL[i] for i in rightL]

    NV = []
    for i in range(len(iN)):
        if i != r:
            NV.append((max([p[i][j] + p[i+len(p)//2][j] for j in rightL])
                - max([p[i][j] + p[i+len(p)//2][j] for j in leftL]),
                i))
    NV.sort()
    leftN = NV[:len(leftL)-1]; rightN = NV[len(leftL)-1:]
    ilN = [iN[i] for _, i in leftN]
    irN = [iN[i] for _, i in rightN]
    pl  = [[p[i][j] for j in leftL] for _, i in leftN]
    pl += [[p[i+len(p)//2][j] for j in leftL] for _, i in leftN]
    pr  = [[p[i][j] for j in rightL] for _, i in rightN]
    pr += [[p[i+len(p)//2][j] for j in rightL] for _, i in rightN]
    
    leftT  = buildTree(pl, ilN, ilL) if len(ilN) > 0 else ilL
    rightT = buildTree(pr, irN, irL) if len(irN) > 0 else irL
    return [iN[r], leftT, rightT]

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def dt2dn(dt, tree, a, b, c, d, n, gamma):
    connectivity = [
        np.zeros((a, b)),
        np.zeros((b, c)),
        np.zeros((c, d))
    ]
    eps = 10**-3
    weight = [
        np.zeros((a, b)),
        np.zeros((b, c)),
        np.zeros((c, d))+log(e**eps-1),
        np.zeros((d, 1))
    ]
    bias = [
        np.zeros(b),
        np.zeros(c),
        np.zeros(d),
        np.zeros(1)
    ]

    nbNodes      = tree.node_count
    father, side = makeTree(tree)
    nodes, leafs = indexNodes(tree), indexLeafs(tree)
    nodeMap      = revIndex(nodes)

    for j, node in enumerate(nodes):
        connectivity[0][tree.feature[node]][j] = 1.
        weight[0][tree.feature[node]][j]       = gamma*1.
        bias[0][j]                             = - gamma * tree.threshold[node]

    for j in range(b):
        connectivity[1][j][j] = 1
        weight[1][j][j]       = 1
        bias[1][j]            = 0

        connectivity[1][j][j+b] = 1
        weight[1][j][j+b]       = -1
        bias[1][j+b]            = 1
    
    for j, leaf in enumerate(leafs):
        v   = side[leaf]
        cur = father[leaf]
        while cur != -1:
            curI = nodeMap[cur]
            if v != 1.:
                curI += b
            connectivity[2][curI][j] = 1.
            weight[2][curI][j]       = log(e-1)

            v    = side[cur]
            cur  = father[cur]

        weight[3][j][0]  = tree.value[leaf] / n

    return connectivity, weight, bias

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

def getRoot(p, balance=False):
    def getExp(x):
        return sum([p[x][i] + p[x+len(p)//2][i] for i in range(len(p[x]))])
    
    best = 0
    bestV = getExp(0)
    delta = len(p[0])+1
    for x in range(len(p)//2):
        if balance:
            L, R = 0, 0
            for i in range(len(p[0])):
                if p[x][i] - p[x+len(p)//2][i] < 0:
                    R += 1
                else:
                    L += 1
            if delta > abs(R-L):
                delta = abs(R-L)
                best = x
            elif delta == abs(R-L):
                if getExp(x) > getExp(best):
                    best = x
        else:
            curV = getExp(x)
            if curV > bestV:
                bestV = curV
                best  = x

    return best

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

def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

def revIndex(index):
    mp = {}
    for i in range(len(index)):
        mp[index[i]] = i
    return mp

def rmse(a, b):
    return sqrt(np.mean(sq(a-b)))

def selectBatch(data, batchSize, replace=True, unzip=True, rs=None):
    if rs is None:
        rs = RandomState(rs)
    ret = []
    for i in range(batchSize):
        j = rs.randint(0 if replace else i, len(data)-1)
        ret.append(data[j])
        if not replace:
            data[i], data[j] = data[j], data[i]
    return map(list, zip(* ret)) if unzip else ret

def sq(x):
    return x*x

def zipData(data):
    x, y = map(list, zip(* data))
    return x, [yy[0] for yy in y]

