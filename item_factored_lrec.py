#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification
from load_movielens import load_data
from rank_metrics import prec, apk, recall
from sklearn.decomposition import PCA


def main():
    ks = [3, 5, 10, 20]
    mapk = 200
    train, test = load_data()
    train, test = train.as_matrix(), test.as_matrix()
    x = train
    pca = PCA(n_components=int(train.shape[1]*0.01))
    pca.fit(train)
    new_x = pca.transform(train)
    res = []
    for i in xrange(train.shape[1]):
        y = x[:, i]
        clf = LogisticRegression(random_state=42, C=0.001, solver='lbfgs')
        clf.fit(new_x, y)
        pred_buy_proba = clf.predict_proba(new_x)[:,1].ravel()
        res.append(pred_buy_proba)
    res = np.array(res).T
    pred = (res - train).argsort(axis=1)[::-1]

    res = np.zeros(9)
    for u in xrange(train.shape[0]):
        truth = test[u]
        pred_order = pred[u]
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
    main()
