from sklearn import metrics
import heapq
import numpy as np
import math


def calculate_ndcg(*buffer):
    top_items = heapq.nlargest(buffer[0], list(zip(buffer[1], buffer[2])))
    num_postive = int(sum(buffer[2]))

    dcg = 0
    idcg = 0
    for i, (score, label) in enumerate(top_items):
        if label == 1:
            dcg += math.log(2) / math.log(i + 2)

        if i < num_postive:
            idcg += math.log(2) / math.log(i + 2)

    return dcg / idcg


def calculate_auc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), prob, pos_label=1)
    auc = float(metrics.auc(fpr, tpr))

    return auc
