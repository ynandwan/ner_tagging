#ref - https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#evaluation


import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV 

train_file = '../data/train_crf.pkl'
num_groups = 10
train_data = pickle.load(open(train_file,'rb'))


np.random.seed(25120)
groups = np.random.randint(num_groups, size = len(train_data))


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


X = [sent2features(s) for s in train_data]
Y = [sent2labels(s) for s in train_data]

cross_val_results = []
for gid in range(num_groups):
    X_train = [X[i] for i in range(len(X)) if groups[i] != gid]
    y_train = [Y[i] for i in range(len(Y)) if groups[i] != gid]
    
    X_test = [X[i] for i in range(len(X)) if groups[i] == gid]
    y_test = [Y[i] for i in range(len(Y)) if groups[i] == gid]
    
    %%time
    crf = sklearn_crfsuite.CRF(
        algorithm='pa',
        c=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    labels.remove('O')
    labels
    y_pred = crf.predict(X_test)
    cross_val_results.append(metrics.flat_f1_score(y_test, y_pred,
                average='macro', labels=labels))
    
np.mean(cross_val_results)


def grid_search(X, y, labels):
    crf = sklearn_crfsuite.CRF(
        algorithm='pa',
        max_iterations=100,
        all_possible_transitions=False
    )
    params_space = {
        'c': [0.1]
    }


    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='macro', labels=labels)

    # search
    rs = GridSearchCV(crf, params_space,
                            cv=10,
                            verbose=1,
                            n_jobs=-1,
                            # n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X, y)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)         # c1=0.5, c2=0.55
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    return rs


grid_search(X,Y, ['D','T'])
