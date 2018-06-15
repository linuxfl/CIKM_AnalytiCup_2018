#coding:utf-8
import sys
import math
import numpy as np
import pandas as pd
import jieba, gensim
import xgboost as xgb
from tqdm import tqdm
import scipy

from scipy.stats import skew, kurtosis
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel
basic_feature = ['len_word_s1', 'len_word_s2', 'len_char_s2', 'len_char_s1', 'len_ratio']
fuzz_feature = ['fuzz_QRatio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
gramoverlap_feature = ['1-gramoverlap', '2-gramoverlap', '3-gramoverlap', '2-gramoverlap_char', '3-gramoverlap_char', '4-gramoverlap_char', '5-gramoverlap_char']
other_feature = ['bow', 'bow_tfidf', 'lcs_diff']
sequence_feature = ['long_common_sequence', 'long_common_prefix', 'long_common_suffix', 'long_common_substring', 'long_common_substring']
#, 'levenshtein_distance']
word2vec_feature = ['cityblock_distance', 'jaccard_distance', 'minkowski_distance', 'skew_s1vec', 'skew_s2vec', 'kur_s1vec', 'kur_s2vec', 'pearson_coff', 'spearman_coff', 'kendalltau_coff']
#, 'polynomial_kernel', 'rbf_kernel',  'laplacian_kernel']

feature = []
feature.extend(basic_feature)
feature.extend(fuzz_feature)
feature.extend(gramoverlap_feature)
feature.extend(other_feature)
feature.extend(word2vec_feature)
feature.extend(sequence_feature)

words_dict = {}
with open("words.txt") as fin:
    for raw_line in fin:
        line = raw_line.strip("\n\r").split()
        if len(line) != 2:
            continue
        words_dict[line[0]] = float(line[1])

word2vec_dict = {}
with open("word2vec.dict") as fin:
    for raw_line in fin:
        line = raw_line.strip("\n\r").split()
        word2vec_dict[line[0]] = [float(n) for n in line[1:]]

def shuffle(X, y):
   m = X.shape[0]
   ind = np.arange(m)
   for i in range(7):
       np.random.shuffle(ind)
   return X[ind], y[ind]

def euclidean(vec1, vec2):
    num = len(vec1)
    s = 0.0
    for i in range(num):
        s += (vec1[i] - vec2[i])**2
    return math.sqrt(s)

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

def bag_of_words_tfidf(sentences):
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
            try:
                if i == 0:
                    sen_vec_1[words[word]] = words_dict[word]
                else:
                    sen_vec_2[words[word]] = words_dict[word]
            except:
                if i == 0:
                    sen_vec_1[words[word]] = 0
                else:
                    sen_vec_2[words[word]] = 0
                
    return cos(sen_vec_1, sen_vec_2)
   
def lcs_diff(sentences):
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
def long_common_sequence(sentences):
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

def long_common_prefix(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)

    max_prefix = 0
    min_len = min(len1, len2)
    for i in range(min_len):
        if sen1[i] == sen2[i]:
            max_prefix += 1

    return max_prefix * 1.0 / (len1 + len2)

def long_common_suffix(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)

    min_len = min(len1, len2)
    max_suffix = 0
    for i in range(min_len, 0, 1):
        if sen1[i] == sen2[i]:
            max_suffix += 1

    return max_suffix * 1.0 / (len1 + len2)

def long_common_substring(sentences):
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
                dp[i][j] = 0
    return dp[len1-1][len2-1] * 1.0 / (len1 + len2)

def levenshtein_distance(sentences):
    sen = sentences.split("\001")
    s = sen[0].split(" ")
    t = sen[1].split(" ")

    if s == t: return 0  
    elif len(s) == 0: return len(t)  
    elif len(t) == 0: return len(s)  
    v0 = [None] * (len(t) + 1)  
    v1 = [None] * (len(t) + 1)  
    for i in range(len(v0)):  
        v0[i] = i  
    for i in range(len(s)):  
        v1[0] = i + 1  
        for j in range(len(t)):  
            cost = 0 if s[i] == t[j] else 1  
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)  
        for j in range(len(v0)):  
            v0[j] = v1[j]  

    return v1[len(t)]       

def n_gram_over_lap(sentences, n):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)

    if n == 1:
        word_set1 = set(sen1)
        word_set2 = set(sen2)

        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        return num1 * 1.0 / num2
    elif n == 2:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0:
                continue
            word_set1.add(sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0:
                continue
            word_set2.add(sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        return num1 * 1.0 / num2
    else:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0 or i == 1:
                continue
            word_set1.add(sen1[i-2] + sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0 or i == 1:
                continue
            word_set2.add(sen2[i-2] + sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        if num2 == 0:
            num2 = 1
        return num1 * 1.0 / num2

    return 0

def n_gram_over_lap_char(sentences, n):
    sen = sentences.split("\001")
    sen1 = ''.join(sen[0].split(" "))
    sen2 = ''.join(sen[1].split(" "))
    
    len1 = len(sen1)
    len2 = len(sen2)

    if n == 2:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0:
                continue
            word_set1.add(sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0:
                continue
            word_set2.add(sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        return num1 * 1.0 / num2
    elif n == 3:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0 or i == 1:
                continue
            word_set1.add(sen1[i-2] + sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0:
                continue
            word_set2.add(sen2[i-2] + sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        return num1 * 1.0 / num2
    elif n == 4:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0 or i == 1 or i == 2:
                continue
            word_set1.add(sen1[i-3] + sen1[i-2] + sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0 or i == 1 or i == 2:
                continue
            word_set2.add(sen2[i-3] + sen2[i-2] + sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        if num2 == 0:
            num2 = 1
        return num1 * 1.0 / num2
    else:
        word_set1 = set()
        word_set2 = set()
        for i in range(len1):
            if i == 0 or i == 1 or i == 2 or i == 3:
                continue
            word_set1.add(sen1[i-4] + sen1[i-3] + sen1[i-2] + sen1[i-1] + sen1[i])
        for i in range(len2):
            if i == 0 or i == 1 or i == 2:
                continue
            word_set2.add(sen2[i-4] + sen2[i-3] + sen2[i-2] + sen2[i-1] + sen2[i])
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        if num2 == 0:
            num2 = 1
        return num1 * 1.0 / num2


    return 0

def len_ratio(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")
    len1 = len(sen1)
    len2 = len(sen2)
    if len2 == 0:
        len2 == 1
    return len1 * 1.0 / len2

def get_doc2vec_sim(sentences):
    sen = sentences.split("\001")
    sen1 = sen[0].split(" ")
    sen2 = sen[1].split(" ")

    inferred_vector1 = model.infer_vector(sen1)
    inferred_vector2 = model.infer_vector(sen2)
    return euclidean(inferred_vector1, inferred_vector2)

def read_data(filename):
    return pd.read_csv(filename, names=['label', 'sentences'], sep='\t')

def fuzz_QRatio(sentences):
    sen = sentences.split("\001")
    return fuzz.QRatio(sen[0], sen[1]) 

def fuzz_WRatio(sentences):
    sen = sentences.split("\001")
    return fuzz.WRatio(sen[0], sen[1])

def fuzz_partial_ratio(sentences):
    sen = sentences.split("\001")
    return fuzz.partial_ratio(sen[0], sen[1])

def fuzz_partial_token_set_ratio(sentences):
    sen = sentences.split("\001")
    return fuzz.partial_token_set_ratio(sen[0], sen[1])

def fuzz_partial_token_sort_ratio(sentences):
    sen = sentences.split("\001")
    return fuzz.partial_token_sort_ratio(sen[0], sen[1])

def fuzz_token_set_ratio(sentences):
    sen = sentences.split("\001")
    return fuzz.token_set_ratio(sen[0], sen[1])

def fuzz_token_sort_ratio(sentences):
    sen = sentences.split("\001")
    return fuzz.token_sort_ratio(sen[0], sen[1])

def sent2vec(sen):
    M = []
    words = sen.split()
    for w in words:
        try:
            M.append(word2vec_dict[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v**2).sum())

def generate_feature(data):
    """
        basic feature
    """
    #length of sentence
    data['len_word_s1'] = data.apply(lambda x: len(x['sentences'].split("\001")[0].split(" ")), axis=1)
    data['len_word_s2'] = data.apply(lambda x: len(x['sentences'].split("\001")[1].split(" ")), axis=1)
    data['len_ratio'] = data.apply(lambda x: len_ratio(x['sentences']), axis=1)
    data['len_char_s1'] = data.apply(lambda x: len(''.join(x['sentences'].split("\001")[0].split(" "))), axis=1)
    data['len_char_s2'] = data.apply(lambda x: len(''.join(x['sentences'].split("\001")[1].split(" "))), axis=1)
    """
        fuzzywuzzy feature
    """
    data['fuzz_QRatio'] = data.apply(lambda x: fuzz_QRatio(x['sentences']), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz_WRatio(x['sentences']), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz_partial_ratio(x['sentences']), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz_partial_token_set_ratio(x['sentences']), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz_partial_token_sort_ratio(x['sentences']), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz_token_set_ratio(x['sentences']), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz_token_sort_ratio(x['sentences']), axis=1)
    """
        word2vec feature
    """
    sent1_vectors = np.zeros((data.shape[0], 300))
    for i, sents in tqdm(enumerate(data.sentences.values)):
        sent = sents.split("\001")[0]
        sent1_vectors[i, :] = sent2vec(sent)

    sent2_vectors = np.zeros((data.shape[0], 300))
    for i, sents in tqdm(enumerate(data.sentences.values)):
        sent = sents.split("\001")[1]
        sent2_vectors[i, :] = sent2vec(sent)
    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    data['pearson_coff'] = [scipy.stats.pearsonr(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    data['spearman_coff'] = [scipy.stats.spearmanr(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    data['kendalltau_coff'] = [scipy.stats.kendalltau(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    
    #data['polynomial_kernel'] = [polynomial_kernel(x.reshape(1, -1), y.reshape(1, -1)) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['sigmoid_kernel'] = [sigmoid_kernel(x.reshape(1, -1), y.reshape(-1, 1)) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['rbf_kernel'] = [rbf_kernel(x.reshape(-1, 1), y.reshape(-1, 1)) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
    #data['laplacian_kernel'] = [laplacian_kernel(x.reshape(-1, 1), y.reshape(-1, 1)) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]

    data['skew_s1vec'] = [skew(x) for x in np.nan_to_num(sent1_vectors)]
    data['skew_s2vec'] = [skew(x) for x in np.nan_to_num(sent2_vectors)]
    data['kur_s1vec'] = [kurtosis(x) for x in np.nan_to_num(sent1_vectors)]
    data['kur_s2vec'] = [kurtosis(x) for x in np.nan_to_num(sent2_vectors)]
    """
        Sequence Features
    """
    data['long_common_sequence'] = data.apply(lambda x: long_common_sequence(x['sentences']), axis=1)
    data['long_common_prefix'] = data.apply(lambda x: long_common_prefix(x['sentences']), axis=1)
    data['long_common_suffix'] = data.apply(lambda x: long_common_suffix(x['sentences']), axis=1)
    data['long_common_substring'] = data.apply(lambda x: long_common_substring(x['sentences']), axis=1)
    data['levenshtein_distance'] = data.apply(lambda x: levenshtein_distance(x['sentences']), axis=1)

    #Bag-of-Words
    data['bow'] = data.apply(lambda x: bag_of_words(x['sentences']), axis=1)
    data['bow_tfidf'] = data.apply(lambda x: bag_of_words_tfidf(x['sentences']), axis=1)
    data['lcs_diff'] = data.apply(lambda x: lcs_diff(x['sentences']), axis=1)
    #N-gramOverlap
    data['1-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 1), axis=1)
    data['2-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 2), axis=1)
    data['3-gramoverlap'] = data.apply(lambda x: n_gram_over_lap(x['sentences'], 3), axis=1)
    data['2-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 2), axis=1)
    data['3-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 3), axis=1)
    data['4-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 4), axis=1)
    data['5-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 5), axis=1)
    return data
 

def train_and_predict(X_train, Y_train, X_test):
    #XX_train, XX_test, YY_train, YY_test = train_test_split(X_train, Y_train, random_state=2018)
    """
    lr_model = LogisticRegression()
    lr_model.fit(XX_train, YY_train)
    pred = lr_model.predict_proba(XX_test)[:,1]
    print "lr test logloss is {}".format(metrics.log_loss(YY_test, pred))
    X_train, Y_train = shuffle(X_train, Y_train)
    lr_model.fit(X_train, Y_train)
    pred = lr_model.predict_proba(X_test)[:,1]
    """ 
    gbdt_model = GradientBoostingClassifier(n_estimators=150, max_depth=6, loss="deviance")
    logloss = cross_val_score(gbdt_model, X_train, Y_train, cv=5, scoring='neg_log_loss')
    print -logloss.mean()
    #print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred1))
    #gbdt_model.fit(X_train, Y_train)
    #pred = gbdt_model.predict_proba(X_test)[:,1]
    #np.savetxt("result.txt", pred, fmt='%1.6f')
    """
    rf_model = RandomForestClassifier(n_estimators=500)
    logloss = cross_val_score(rf_model, X_train, Y_train, cv=10, scoring='neg_log_loss')
    print -logloss.mean()
    rf_model.fit(X_train, Y_train)
    pred = rf_model.predict_proba(X_test)[:,1]
    np.savetxt("result.txt", pred, fmt='%1.6f')
    dtrain = xgb.DMatrix(XX_train, label=YY_train)
    dtest = xgb.DMatrix(XX_test, label=YY_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {'max_depth': 5, 'lambda':1.1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'logloss'
    bst = xgb.train(param, dtrain, 150, evallist)
    pred = bst.predict(dtest)
    print "gbdt test logloss is {}".format(metrics.log_loss(YY_test, pred))
    """
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
