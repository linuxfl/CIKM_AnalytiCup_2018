#coding:utf-8
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import jieba,gensim

def shuffle(X, y):
   m = X.shape[0]
   for i in range(7):
       ind = np.arange(m)
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
    X_train[:,1:150] = X_train[:,1:150] - X_train[:,150:300]
    X_test[:,1:150] = X_test[:,1:150] - X_test[:,150:300]
    X_train = X_train[:,0:150]
    X_test = X_test[:,0:150]
    print X_train.shape, X_test.shape, Y_train.shape
    XX_train, XX_test, YY_train, YY_test = train_test_split(X_train, Y_train, random_state=0)
    lr_model = LogisticRegression()
    lr_model.fit(XX_train, YY_train)
    pred = lr_model.predict_proba(XX_test)[:,1]
    print "lr test logloss is {}".format(metrics.log_loss(YY_test, pred))

    #gbdt_model = GradientBoostingClassifier(loss="deviance")
    #gbdt_model.fit(XX_train, YY_train)
    #pred = gbdt_model.predict_proba(XX_test)[:,1]
    #print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))

    #mlp_model = MLPClassifier(solver='adam', shuffle=True, alpha=1e-5, hidden_layer_sizes=(1000, 100,100), random_state=1, activation='relu')
    #mlp_model.fit(XX_train, YY_train)
    #pred = mlp_model.predict_proba(XX_test)[:,1]
    #print "mlp test logloss is {}".format(metrics.log_loss(YY_test, pred))

    lr_model.fit(X_train, Y_train)
    
    pred = lr_model.predict_proba(X_test)[:,1]
    np.savetxt("result.txt", pred, fmt='%1.6f')

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
