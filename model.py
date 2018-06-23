from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import metrics
import numpy as np
import numpy
basic_feature = ['len_word_s1', 'len_word_s2', 'len_char_s2', 'len_char_s1', 'len_ratio']
fuzz_feature = ['fuzz_QRatio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
gramoverlap_feature = ['1-gramoverlap_word', '2-gramoverlap_word', '3-gramoverlap_word', '2-gramoverlap_char', '3-gramoverlap_char', '4-gramoverlap_char', '5-gramoverlap_char']
other_feature = ['bow', 'bow_tfidf', 'lcs_diff', 'has_no_word']
sequence_feature = ['long_common_sequence', 'long_common_substring', 'long_common_suffix', 'long_common_prefix']
#, 'levenshtein_distance']
word2vec_feature_ave_idf = ['cosine_distance_ave_idf', 'canberra_distance_ave_idf', 'jaccard_distance_ave_idf', 'minkowski_distance_ave_idf', 'skew_s1vec_ave_idf', 'skew_s2vec_ave_idf', 'kur_s1vec_ave_idf', 'kur_s2vec_ave_idf', 'kendalltau_coff_ave_idf']

feature = []
feature.extend(basic_feature) #0.6479294163387835
feature.extend(fuzz_feature) #0.48524011604707845
feature.extend(gramoverlap_feature) #0.5050180567638318
feature.extend(sequence_feature) #0.6085519713571158
feature.extend(other_feature) #0.5387043318754241
feature.extend(word2vec_feature_ave_idf)

print "%s features"%(len(feature))
def shuffle(X, y):
   m = X.shape[0]
   ind = np.arange(m)
   for i in range(7):
       np.random.shuffle(ind)
   return X[ind], y[ind]

train_data = pd.read_csv('train.dat')
train_data.fillna(0, inplace=True)
X_test = train_data[feature][train_data.label == -1].values

train_data = train_data[train_data.label >= 0]
X_train = train_data[feature][train_data.label >= 0].values
Y_train = train_data[train_data.label >= 0]['label'].values


rf_model = RandomForestClassifier(n_estimators=600, n_jobs=4)
logloss = cross_val_score(rf_model, X_train, Y_train, cv=5, scoring='neg_log_loss')
print -logloss.mean()

rf_model.fit(X_train, Y_train)
pred = rf_model.predict_proba(X_test)[:,1]
np.savetxt("result.txt", pred, fmt='%1.6f')

gbdt_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, loss="deviance")
logloss = cross_val_score(gbdt_model, X_train, Y_train, cv=5, scoring='neg_log_loss')
print -logloss.mean()

gbdt_model.fit(X_train, Y_train)
pred = gbdt_model.predict_proba(X_test)[:,1]
np.savetxt("result.txt", pred, fmt='%1.6f')


"""
from sklearn import preprocessing
X_train_standard = None
for fea in feature:
    X_train = preprocessing.StandardScaler().fit_transform(train_data[fea].values.reshape(-1, 1))
    if not isinstance(X_train_standard, numpy.ndarray):
        if X_train_standard == None:
            X_train_standard = X_train
    else:
        X_train_standard = np.hstack((X_train_standard, X_train))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=1)
logloss = cross_val_score(clf, X_train_standard, Y_train, cv=5, scoring='neg_log_loss')
print -logloss.mean()

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
logloss = cross_val_score(lr_model, X_train_standard, Y_train, cv=5, scoring='neg_log_loss')
print -logloss.mean()
"""
