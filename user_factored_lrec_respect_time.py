import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification
from load_movielens import load_strict_split_by_time as load_data
from rank_metrics import prec, apk, recall
from sklearn.decomposition import PCA


def main():
    ks = [3, 5, 10, 20]
    mapk = 200
    train, test = load_data()
    train, test = train.as_matrix(), test.as_matrix()
    x = train.T
    res = np.zeros(9)
    number_of_user = train.shape[0]
    print int(number_of_user*0.01)
    pca = PCA(n_components=int(number_of_user*0.01))
    new_x = pca.fit_transform(x)
    for u in xrange(train.shape[0]):
        y = x[:, u]
        truth = test[u]
        clf = LogisticRegression(random_state=42, C=0.001, solver='lbfgs')
        clf.fit(new_x, y)
        #print u, classification.accuracy_score(clf.predict(x), y)
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
    print main(), '!!!'
