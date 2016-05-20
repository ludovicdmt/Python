"""Kaggle competition: Predicting a Biological Response.
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)
The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)
Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.
Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold

def stacking(clfs, X, y, X_submission, n_folds = 5, verbose=True, shuffle = False, random_state = 42):

    np.random.seed(0) # seed to shuffle the train set

    #X, y, X_submission = load_data.load()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds, random_state = 42))

    print ("Creating train and test sets for blending.")
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    scores = np.zeros(len(clfs))
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print ("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
            scores[j] += clf.score(X_test, y_test)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        scores = scores * 1. / n_folds

    return  dataset_blend_train, dataset_blend_test, scores
    
import scipy.stats as stats
def filter_stacking(X, scores, threshold = 0.9, max_col = 5):
    sorted_index = np.argsort(scores)[::-1]
    spearmanr_matrix =  stats.spearmanr(dataset_blend_train)[0]
    res = []
    while len(sorted_index) > 0 &  len(res) < max_col:
        res.append(sorted_index[0])
        #sorted_index = sorted_index[1:]
        sorted_index =  np.intersect1d(sorted_index, np.argwhere(spearmanr_matrix[res[-1],:]<threshold).ravel())
    return X[:,res], np.array(res)

    #print
    #print "Blending."
    
    #clf = LogisticRegression()
    #clf.fit(dataset_blend_train, y)
    #y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    #print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    #print "Saving Results."
    #np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')