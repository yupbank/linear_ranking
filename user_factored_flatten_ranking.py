#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification
from load_movielens import load_data
from rank_metrics import prec, apk, recall


def main():
    ks = [3, 5, 10, 20]
    mapk = 200
    epoch = 1
    res = np.zeros(9)
    train, test = load_data()
    train, test = train.as_matrix(), test.as_matrix()
    number_of_user = train.shape[0]
    print int(number_of_user*0.01)
    pca = PCA(n_components=int(number_of_user*0.01))

    x = train.T
    clf = SGDClassifier(random_state=42, loss='log')

    factored_x = pca.fit_transform(x)
    factored_y = pca.fit_transform(x.T)

    for i in xrange(epoch):
        for u in xrange(train.shape[0]):
            y = x[:, u]
            new_x = np.append(factored_x, np.repeat(factored_y[u][np.newaxis,:], x.shape[0], axis=0), axis=1)
            clf.partial_fit(new_x, y, classes=[0, 1])
            if u > 10:
                break

    for u in xrange(train.shape[0]): 
        y = x[:, u]
        truth = test[u]
        #new_x = np.append(x, np.repeat(y[:, np.newaxis], x.shape[1], axis=1), axis=1)
        new_x = np.append(factored_x, np.repeat(factored_y[u][np.newaxis,:], x.shape[0], axis=0), axis=1)
        clf.predict(new_x)
        pred_buy_proba = clf.predict_proba(new_x)[:,1].ravel()
        pruned_buy_proba = pred_buy_proba - y.ravel()
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
    main()
