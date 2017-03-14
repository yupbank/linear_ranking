#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

from sklearn.linear_model import LogisticRegression
from load_movielens import load_data
from rank_metrics import precision_at_k, average_precision
import numpy as np


def main():
    ks = [3, 5, 10, 20]
    mapk = 200
    train, test = load_data()
    train, test = train.as_matrix(), test.as_matrix()
    x = train.T
    res = np.zeros(5)
    for u in xrange(train.shape[0]):
    #for u in xrange(4):
        y = x[:, u]
        truth = test[u]
        clf = LogisticRegression(C=0.001)
        clf.fit(x, y)
        pred_buy_proba = clf.predict_proba(x)[:,1].ravel()
        pruned_buy_proba = pred_buy_proba - y.ravel()
        pred_order = pruned_buy_proba.argsort()[::-1]
        r = truth[pred_order]
        score= average_precision(r[:mapk])
        tmp = [score]
        for k in ks:
            tmp.append(precision_at_k(r, k))
        res += np.array(tmp)
        if u%50 == 0:
            print res/(u+1)
    return res/(u+1)
if __name__ == "__main__":
    main()
