"""
Model optimisation.
@author Yann Carbonne / modified by Aurelien Galicher
Use of hyperopt module to optimise a XGBoost model.
Great tutorial: http://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
Store each predictions in folder hyperopt_preds/
"""

from preprocessing import load_preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK
#import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier 



def eval_param(params):
    """Evaluation of one set of xgboost's params.
    Then, use 3 folds as training and cv in a row as xgboost's watchlist with an early_stop at 50.
    """
    global df_results, train, target, test
    print ("Training with params : ")
    print (params)

    random_state = 42
    avg_score = 0.
    n_folds = 3
    predict = np.zeros(test.shape[0])
    #dtest = xgb.DMatrix(test)
    skf = StratifiedKFold(target, n_folds=n_folds, random_state=random_state)
    for train_index, cv_index in skf:
        # train
        x_train, x_cv = train[train_index], train[cv_index]
        y_train, y_cv = target[train_index], target[cv_index]
        #dtrain = xgb.DMatrix(x_train, label=y_train)
        #dvalid = xgb.DMatrix(x_cv, label=y_cv)
        #watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        clf = RandomForestClassifier(**params).fit(x_train, y_train)
        #bst = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds, maximize=True)
            # test / score
        predict_cv = clf.predict_proba(x_cv)[:,1]
        avg_score += -log_loss(y_cv, predict_cv)
        predict += clf.predict_proba(test)[:,1]#bst.predict(dtest, ntree_limit=bst.best_iteration)
    predict /= n_folds
    avg_score /= n_folds 
    # store
    new_row = pd.DataFrame([np.append([avg_score], list(params.values()))],
                                 columns=np.append(['score'], list(params.keys())))
    df_results = df_results.append(new_row, ignore_index=True)
    #np.savetxt('hyperopt_preds/pred' + str(df_results.index.max()) + '.txt', predict, fmt='%s')
    df_results.to_csv('hyperopt_results_rf.csv')
    print ("\tScore {0}\n\n".format(avg_score))
    return {'loss': - avg_score, 'status': STATUS_OK}


if __name__ == '__main__':
    train, target, test, _, _, _, _ = load_preprocessing()

    space = {
         'max_depth': hp.quniform('max_depth', 8, 20, 1),
         'max_features': hp.choice('max_features', ['auto','log2', None]),
         'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 6, 1),
         'n_estimators': hp.choice('n_estimators', range(100, 500, 50))
         'criterion': hp.choice('criterion', ["gini", "entropy"]),
         'n_jobs' : -1,
    }
    df_results = pd.DataFrame(columns=np.append(['score'], list(space.keys())))
    best = fmin(eval_param, space, algo=tpe.suggest, max_evals=300)

    print (best)


    best_param_df = pd.read_csv('hyperopt_results_sgd.csv')
    best_param = best_param_df.sort_values('score', ascending=False).head(1).to_dict(orient='records')[0]
    del best_param['Unnamed: 0']
    del best_param['score']
    # Send a text message with twillio
    #from my_phone import send_text
    #send_text("Best params: " + str(best))