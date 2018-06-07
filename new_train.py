#coding:utf-8
import sys
import math
import numpy as np
import pandas as pd
import jieba, gensim
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import metrics

feature = ['bow', 'lcs', 'postag', 'lenratio']
def shuffle(X, y):
   m = X.shape[0]
   ind = np.arange(m)
   for i in range(7):
       np.random.shuffle(ind)
   return X[ind], y[ind]

def cos(vec1, vec2):
    length = len(vec1)
    num = 0.0
    vec1_norm = 0.0
    vec2_norm = 0.0
    for i in range(length):
        num += vec1[i] * vec2[i]
        vec1_norm += vec1[i] * vec1[i]
        vec2_norm += vec2[i] * vec2[i]
    norm = math.sqrt(vec1_norm) * math.sqrt(vec2_norm)
    return num / norm

#bag of words feature
def bag_of_words(sentences):
    words = {}
    pos = 0
    for sen in sentences.split("\001"):
        for word in sen.split():
            if word not in words:
                words[word] = pos
                pos += 1
    sen_vec_1 = [ 0 for i in range(pos) ]
    sen_vec_2 = [ 0 for i in range(pos) ]
    sens = sentences.split("\001")
    for i, sen in enumerate(sens):
        for word in sen.split():
            if i == 0:
                sen_vec_1[words[word]] = 1
            else:
                sen_vec_2[words[word]] = 1
    return cos(sen_vec_1, sen_vec_2)

#pos tags (nomalize)
def lcs_pos(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)

    dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
    offset = 0
    for i in range(len1):
        for j in range(len2):
            if sen1[i] == sen2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
                of = abs(j - i)
                offset += of
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return offset * 1.0 / (len1 + len2)

#longest common subsequence 
def average_lcs(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)

    dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
    for i in range(len1):
        for j in range(len2):
            if sen1[i] == sen2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[len1-1][len2-1] * 1.0 / (len1 + len2)

def n_gram_over_lap(sentences, n):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)
    
    return 0

def len_ratio(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)
    if len2 == 0:
        len2 == 1
    #return abs(len1 - len2) * 1.0 / max(len2, len1)
    #return min(len1, len2) * 1.0 / max(len2, len1)
    return len1 * 1.0 / len2

def read_data(filename):
    return pd.read_csv(filename, names=['label', 'sentences'], sep='\t')

def generate_feature(data):
    #Bag-of-Words
    data['bow'] = data.apply(lambda x: bag_of_words(x['sentences']), axis=1)
    #POS Tags Longest Common Subsequence
    data['lcs'] = data.apply(lambda x: average_lcs(x['sentences']), axis=1)
    data['postag'] = data.apply(lambda x: lcs_pos(x['sentences']), axis=1)
    #data['postag'] = data.apply(lambda x: standard_scaler(x), axis=1)
    #N-gramOverlap
    data['1-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 1), axis=1)
    data['2-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 2), axis=1)
    data['3-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 3), axis=1)
    #length ratio
    data['lenratio'] = data.apply(lambda x: len_ratio(x['sentences']), axis=1)

    #standard_scaler
    #scaler = preprocessing.StandardScaler()
    #data['norpostag'] = pd.Series(scaler.fit_transform(data['postag'].values.reshape(-1, 1)).reshape(-1))
    #data['norlenratio'] = pd.Series(scaler.fit_transform(data['lenratio'].values.reshape(-1, 1)).reshape(-1))
    return data

def train_and_predict(X_train, Y_train, X_test):
    XX_train, XX_test, YY_train, YY_test = train_test_split(X_train, Y_train, random_state=0)
    lr_model = LogisticRegression()
    lr_model.fit(XX_train, YY_train)
    pred = lr_model.predict_proba(XX_test)[:,1]
    print "lr test logloss is {}".format(metrics.log_loss(YY_test, pred))
    X_train, Y_train = shuffle(X_train, Y_train)
    lr_model.fit(X_train, Y_train)
    pred = lr_model.predict_proba(X_test)[:,1]

    gbdt_model = GradientBoostingClassifier(n_estimators=100, max_depth=8, loss="deviance")
    gbdt_model.fit(XX_train, YY_train)
    pred = gbdt_model.predict_proba(XX_test)[:,1]
    print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))
    #gbdt_model.fit(X_train, Y_train)
    #pred = gbdt_model.predict_proba(X_test)[:,1]
    
    #dtrain = xgb.DMatrix(XX_train, label=YY_train)
    #dtest = xgb.DMatrix(XX_test, label=YY_test)
    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
    #param = {'max_depth': 3, 'eta': 1.5, 'silent': 1, 'objective': 'binary:logistic'}
    #param['nthread'] = 4
    #param['eval_metric'] = 'logloss'
    #bst = xgb.train(param, dtrain, 50, evallist)
    #pred = bst.predict(dtest)
    #print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))
    
    np.savetxt("result.txt", pred, fmt='%1.6f')

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print "Usage: train.py train_ins"
        exit()
    data = read_data(sys.argv[1])
    train_data = generate_feature(data)

    X_train = train_data[feature][train_data.label >= 0].values
    y_train = train_data[train_data.label >= 0]['label'].values
    X_test = train_data[feature][train_data.label == -1].values
    print X_train.shape, y_train.shape, X_test.shape
    train_and_predict(X_train, y_train, X_test)
