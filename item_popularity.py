#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
from sklearn.metrics import classification
from load_movielens import load_data
from rank_metrics import prec, apk, recall


def main():
    ks = [3, 5, 10, 20]
    mapk = 200
    train, test = load_data()
    train, test = train.as_matrix(), test.as_matrix()
    pred = train.sum(axis=0)
    res = np.zeros(9)
    x = train.T
    for u in xrange(train.shape[0]):
        y = x[:, u]
        truth = test[u]
        pred_buy_proba = pred
        y[y>0] = float('-inf')
        pruned_buy_proba = pred_buy_proba + y.ravel()
        pred_order = pruned_buy_proba.argsort()[::-1]
        actual_bought = truth.nonzero()[0]
        score= apk(actual_bought, pred_order, mapk)
        tmp = [score]
        for k in ks:
            tmp.append(prec(actual_bought, pred_order, k))
            tmp.append(recall(actual_bought, pred_order, k))
        res += np.array(tmp)
        if u%50 == 0:
            print res/(u+1)
    return res/(u+1)

if __name__ == "__main__":
    print main()
