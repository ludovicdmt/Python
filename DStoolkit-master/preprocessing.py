"""
Preprocessing script.
@author Aurelien Galicher for BNP kaggle Challenge
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import calendar
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
#%matplotlib inline
#import matplotlib.pyplot as plt;
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def load_preprocessing(with_selector=False):
    """Load preprocessed data with or without the selector mask."""
    train, target, test = np.loadtxt('train.txt'), np.loadtxt('target.txt'), np.loadtxt('test.txt')
    xtrain_mix = load_sparse_csr('xtrain_mix.npz')
    xtest_mix = load_sparse_csr('xtest_mix.npz')
    xtrain_cat = load_sparse_csr('xtrain_cat.npz')
    xtest_cat = load_sparse_csr('xtest_cat.npz')
    
    if with_selector:
        selector = np.loadtxt('feature_selector_support.txt')
        selector = np.array(selector, dtype=bool)
        train, test = train[:, selector], test[:, selector]
    return train, target, test, xtrain_mix, xtest_mix, xtrain_cat, xtest_cat


def create_preprocessing():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    submit = pd.read_csv('../data/sample_submission.csv')
    df_all = pd.concat([train, test], axis=0)

    #################
    # Creatind additional features based on NA values
    #################
    na_columns = map(lambda x: x+'_na', df_all.columns)
    df_all_na = df_all.isnull().astype(int)
    df_all_na.columns = na_columns
    df_all = pd.concat([df_all, df_all_na], axis=1)
    
    #################
    # Sorting categorial and continous features.
    #################
    cat_features= train.select_dtypes(include=['object']).columns
    cont_features = train.select_dtypes(exclude=['object']).columns[2:] #removing id & target

    ##################
    # handling missing values
    # imputer NaN with most_frequent value for categorial features
    ##################

    most_frequent_cat_value = {}
    for feature in cat_features:
        most_frequent_cat_value[feature] = df_all[feature].value_counts().index[0]
    #print most_frequent_cat_value
    for feature in cat_features:
        df_all[feature] = df_all[feature].fillna(value=most_frequent_cat_value[feature])

    ##################
    # handling missing values
    # imputer mean for continuous features
    ##################

    imp = Imputer()
    df_all[cont_features] = imp.fit_transform(df_all[cont_features])


    #######################
    # "simple" label encoding of categorial features
    #######################
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(df_all[cat_features]).ravel())
    df_all[cat_features] = le.transform(df_all[cat_features])

    
    #######################
    # "onehotencoding"  encoding of categorial label encoded features
    #######################
    enc = preprocessing.OneHotEncoder()
    X_cat = enc.fit_transform(df_all[cat_features].values)
    
    
    
    ########################
    # Standard scaling (-mean/std)
    ########################
    scaler = StandardScaler()
    df_all[cont_features.union(cat_features)] = scaler.fit_transform(df_all[cont_features.union(cat_features)])

    
    ########################
    # binning numerical features using decison tree
    ########################
    #enc_cont = OneHotEncoder()
    X_cont = GBTEncoder(n_estimators=5, max_depth=3, min_samples_leaf=3).fit_transform(df_all[cont_features].values) #.apply(lambda x: binningData(x)[1])
    
    # Add new date related fields
    #print("Adding new fields...")
    #df_all['day_account_created'] = df_all['date_account_created'].dt.weekday
    #df_all['month_account_created'] = df_all['date_account_created'].dt.month
    #df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter
    #df_all['year_account_created'] = df_all['date_account_created'].dt.year
    #df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour
    #df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday
    #df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month
    #df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter
    #df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year
    #df_all['created_less_active'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days
    
    ########################
    # Recreate train / test / target
    train = df_all[df_all.target.notnull()].drop(['ID','target'], axis=1) 
    test = df_all[df_all.target.isnull()].drop(['ID','target'], axis=1) 
    index_test = df_all[df_all.target.isnull()]['ID'].values 
    target = df_all[df_all.target.notnull()].target.values
    x_train_mix = hstack([train[cont_features].values, X_cat.tocsr()[train.index.values]])
    x_test_mix = hstack([test[cont_features].values, X_cat.tocsr()[test.index.values]])
    x_train_cat = hstack([X_cont.tocsr()[train.index.values], X_cat.tocsr()[train.index.values]])
    x_test_cat = hstack([X_cont.tocsr()[test.index.values], X_cat.tocsr()[test.index.values]])
    
    # Save
    np.savetxt('train.txt', train, fmt='%s')
    np.savetxt('test.txt', test, fmt='%s')
    np.savetxt('target.txt', target, fmt='%s')
    save_sparse_csr('xtrain_mix', x_train_mix.tocsr())
    save_sparse_csr('xtest_mix', x_test_mix.tocsr())
    save_sparse_csr('xtrain_cat', x_train_cat.tocsr())
    save_sparse_csr('xtest_cat', x_test_cat.tocsr())
    
    return train, target, test, x_train_mix, x_test_mix, x_train_cat, x_test_cat