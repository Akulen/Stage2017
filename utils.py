import datetime
import random
import re

def custom_iso(clean=''):
    return re.sub('[\.].*', clean, datetime.datetime.now().isoformat())

def selectBatch(data, batchSize, replace=True):
    ret = []
    for i in range(batchSize):
        if replace:
            j = random.randint(0, len(data)-1)
        else:
            j = random.randint(i, len(data)-1)
        ret.append(data[j])
        data[i], data[j] = data[j], data[i]
    return map(list, zip(* data[:batchSize]))

