"""
Model optimisation.
@author Yann Carbonne / modified by Aurelien Galicher
Use of hyperopt module to optimise a XGBoost model.
Great tutorial: http://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
Store each predictions in folder hyperopt_preds/
"""

from preprocessing import get_preproc
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK
import xgboost as xgb
import numpy as np
import pandas as pd


def _get_indices(array, value, index_column=0):
    return np.where(array[:, index_column] == value)[0]


def eval_param(params):
    """Evaluation of one set of xgboost's params.
    Then, use 3 folds as training and cv in a row as xgboost's watchlist with an early_stop at 50.
    """
    global df_results, train, target, test
    print "Training with params : "
    print params

    random_state = 42
    num_round = 5000
    early_stopping_rounds = 50
    avg_score = 0.
    n_folds = 3
    predict = np.zeros(test.shape[0])
    dtest = xgb.DMatrix(test)
    skf = StratifiedKFold(target, n_folds=n_folds, random_state=random_state)
    for train_index, cv_index in skf:
        # train
        x_train, x_cv = train[train_index], train[cv_index]
        y_train, y_cv = target[train_index], target[cv_index]
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_cv, label=y_cv)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds, maximize=True)
            # test / score
        predict_cv = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        avg_score += -log_loss(y_cv, predict_cv)
        predict += bst.predict(dtest, ntree_limit=bst.best_iteration)
    predict /= n_folds
    avg_score /= n_folds 
    # store
    new_row = pd.DataFrame([np.append([avg_score], params.values())], columns=np.append(['score'], params.keys()))
    df_results = df_results.append(new_row, ignore_index=True)
    np.savetxt('hyperopt_preds/pred' + str(df_results.index.max()) + '.txt', predict, fmt='%s')
    df_results.to_csv('hyperopt_results.csv')
    print "\tScore {0}\n\n".format(avg_score)
    return {'loss': - avg_score, 'status': STATUS_OK}


if __name__ == '__main__':
    train, target, test = get_preproc(with_selector=True)

    space = {'eta': hp.quniform('eta', 0.0001, 0.01, 0.0001),
             'max_depth': hp.quniform('max_depth', 8, 25, 1),
             'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
             'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
             'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
             'lambda': hp.quniform('lambda', 0.01, 1.5, 0.01),
             'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'objective': 'binary:logistic',
             'eval_metric': 'logloss',
             'silent': 1
             }
    df_results = pd.DataFrame(columns=np.append(['score'], space.keys()))
    best = fmin(eval_param, space, algo=tpe.suggest, max_evals=300)

    print best

    # Send a text message with twillio
    #from my_phone import send_text
    #send_text("Best params: " + str(best))