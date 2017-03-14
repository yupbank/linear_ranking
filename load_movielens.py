import os
import pandas as pd
import urllib2
import zipfile
import StringIO
import numpy as np
from datetime import datetime


url1 = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
url2 = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

def download_file(url):
    path = url.split('/')[-1].split('.')[0]
    if not os.path.exists(path):
        response =  urllib2.urlopen(url)
        print 'downloading file...'
        with zipfile.ZipFile(StringIO.StringIO(response.read())) as zf:
            zf.extractall()

    return '%s/ratings.dat'%path

def load_raw_data(url=url1):
    return download_file(url)

def rating_matrix(df):
    matrix = df.pivot(index='uid', columns='iid', values='ratings')
    return matrix.fillna(0)

def ensure_same_shape(train, test):
    similar_user = set(train.index).intersection(set(test.index))
    similar_item = set(train.columns).intersection(set(test.columns))
    similar_user = sorted(similar_user)
    similar_item = sorted(similar_item)
    train = train.loc[similar_user, similar_item]
    test = test.loc[similar_user, similar_item]
    if all(train.sum(axis=1) > 0) and all(test.sum(axis=1) > 0)\
        and all(train.sum(axis=0)>0) and all(test.sum(axis=0)>0):
        return train, test

    train = train[train.sum(axis=1) > 0]
    test = test[test.sum(axis=1) > 0]
    train = train.loc[:, train.sum(axis=0) > 0]
    test = test.loc[:, test.sum(axis=0) > 0]
    return ensure_same_shape(train, test)

def n_folder(df, n=1, frac=0.8):
    data_size, _ = df.shape
    for i in xrange(n):
        yield np.split(df.sample(frac=1, random_state=i), [int(data_size*frac)])

def load_df():
    df = pd.read_csv(load_raw_data(), sep='::', engine='python')
    df.columns = ['uid', 'iid', 'ratings', 'time']
    df = df[df['ratings']>3]
    df['time'] = df['time'].apply(datetime.fromtimestamp)
    df = df[['uid', 'iid', 'ratings', 'time']]
    df['ratings'] = 1
    return df


def load_data():
    df = load_df()
    train, test = n_folder(df).next() 
    train, test = map(rating_matrix, [train, test])
    train, test = ensure_same_shape(train, test)
    return train, test


def load_leave_one_out():
    df = load_df()
    grouped = df.sort_values(['uid', 'time']).groupby('uid')
    train = grouped.apply(lambda x: x.iloc[:-1])
    test = grouped.tail(1)
    train, test = map(rating_matrix, [train, test])
    train, test = ensure_same_shape(train, test)
    return train, test


def main():
    load_raw_data(url1)


if __name__ == "__main__":
    main()
        
