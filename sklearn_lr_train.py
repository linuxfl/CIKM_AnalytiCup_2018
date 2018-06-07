#coding:utf-8
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import jieba,gensim
import xgboost as xgb

def shuffle(X, y):
   m = X.shape[0]
   ind = np.arange(m)
   for i in range(7):
       np.random.shuffle(ind)
   return X[ind], y[ind]

def read_data(filename):
    X_train = []
    Y_train = []
    X_test = []
    with open(filename) as fin:
        for raw_line in fin:
            line = raw_line.strip(" \n\r").split(" ")
            if int(line[0]) == -2:
                continue
            if int(line[0]) != -1:
                Y_train.append(int(line[0]))
                x = []
                for l in line[2:]:
                    x.append(float(l))
                X_train.append(x)
            else:
                x = []
                for l in line[2:]:
                    x.append(float(l))
                X_test.append(x)
    return np.array(X_train), np.array(Y_train), np.array(X_test)

def train_and_predict(X_train, Y_train, X_test):
    XX_train, XX_test, YY_train, YY_test = train_test_split(X_train, Y_train, random_state=0)
    lr_model = LogisticRegression()
    lr_model.fit(XX_train, YY_train)
    pred = lr_model.predict_proba(XX_test)[:,1]
    print "lr test logloss is {}".format(metrics.log_loss(YY_test, pred))

    #gbdt_model = GradientBoostingClassifier(n_estimators=100, max_depth=8, loss="deviance")
    #gbdt_model.fit(XX_train, YY_train)
    #pred = gbdt_model.predict_proba(XX_test)[:,1]
    #print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))

    #gbdt_model.fit(X_train, Y_train)
    #pred = gbdt_model.predict_proba(X_test)[:,1]
    dtrain = xgb.DMatrix(XX_train, label=YY_train)
    dtest = xgb.DMatrix(XX_test, label=YY_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth': 2, 'eta': 1.5, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'logloss'
    bst = xgb.train(param, dtrain, 50, evallist)
    pred = bst.predict(dtest)
    print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))
    #np.savetxt("result.txt", pred, fmt='%1.6f')

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print "Usage: train.py train_ins"
        exit()

    print "Read Data From File ..."
    X, y, y1 = read_data(sys.argv[1])
    print "Shuffle Train Instance ..."
    X, y = shuffle(X, y)
    print "LR Train and Score ..."
    train_and_predict(X, y, y1)
