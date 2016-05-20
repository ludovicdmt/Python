
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sparse

def featTransformUsingGBT(clf, x, enc=None):
		x_cat = np.zeros((clf.n_estimators, x.shape[0]))
		X = np.swapaxes(np.array([x]),0,1)
		for i, dect in enumerate(clf.estimators_):
			#x_cat[i] = i*np.ones(x.shape[0])
			x_cat[i] =  dect[0].apply(X)
		if enc is None:
			one_hot_enc = OneHotEncoder(handle_unknown='ignore')
			Xcat = one_hot_enc.fit_transform(x_cat.T)
		else:
			one_hot_enc = enc
			Xcat = one_hot_enc.transform(x_cat.T)
		return Xcat, one_hot_enc

class GBTEncoder(BaseEstimator, TransformerMixin):
	"""Encode labels with GBT"""
	def __init__(self, n_estimators=5, max_depth=5, min_samples_leaf=1):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.gbt_clfs = None
		self.oh_encoders = None
		self.p = 1;
		self.X = None

	def fit(self, X, y=None):
		self.X = X.copy()
		self.p = X.shape[1]
		self.gbt_clfs = []
		self.oh_encoders = []
		### fitting GBT
		for i in range(self.p):
			clf = GradientBoostingRegressor(n_estimators=self.n_estimators,
        		max_depth= self.max_depth, min_samples_leaf= self.min_samples_leaf)
			X_ = np.swapaxes(np.array([X[:,i]]),0,1)
			y_ = X[:,i]
			clf.fit(X_,y_)
			self.gbt_clfs.append(clf)
		### fitting Encoding
		for i,clf in enumerate(self.gbt_clfs):
			_, oh_enc = featTransformUsingGBT(clf, X[:,i])
			self.oh_encoders.append(oh_enc)
		return self

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)

	def transform(self, X):
		Xcat = None
		for i,clf in enumerate(self.gbt_clfs):
			if i > 0:
				Xcat = sparse.hstack([Xcat, featTransformUsingGBT(clf, X[:,i]
					,enc= self.oh_encoders[i])[0]])
			else:
				Xcat = featTransformUsingGBT(clf, X[:,i]
					,enc= self.oh_encoders[i])[0]
		return Xcat

	def inverse_transform(self, X):
		print ("not implemented")
		return None