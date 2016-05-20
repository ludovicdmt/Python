"""
Final prediction.
@author Yann Carbonne.
Give 0.7212 on the public leaderboard.
"""
from preprocessing import get_preproc
import pandas as pd
import xgboost as xgb
import numpy as np


# get data
train, target, test = get_preproc(with_selector=True)

# get best param
best_param_df = pd.read_csv('hyperopt_results.csv')
best_param = best_param_df.sort_values('score', ascending=False).head(1).to_dict(orient='records')[0]
del best_param['Unnamed: 0']
del best_param['score']
num_rounds = 5000
# train & predict
dtrain = xgb.DMatrix(train, label=target)
dtest = xgb.DMatrix(test)
num_rounds = 5000
bst = xgb.train(best_param, dtrain, num_rounds)
y_predict = bst.predict(dtest)

''' writing prediction to file'''
submit = pd.read_csv('sample_submission.csv')
df_test = pd.read_csv('test.csv')
df_submit = pd.DataFrame(columns = submit.columns)
df_submit.ID = df_test.ID 
df_submit.PredictedProb = y_predict
df_submit.to_csv('submission_final.csv', index=False)
