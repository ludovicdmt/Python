from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import scipy.sparse as sparse

def select_random_features(X, sample_size = 0.9, random_state= None):
	if random_state is None:
		rg = np.random.RandomState()
	elif type(random_state) is int:
		rg = np.random.RandomState(seed = random_state)
	else:
		rg = random_state
	p = X.shape[1]
	k = int(sample_size*p)
	#print (k)
	seq = rg.choice(p, k)
	#print (seq)
	if sparse.issparse(X):
		return X.tocsc()[:, seq]
	else:
		return X[:, seq]

from sklearn.preprocessing import FunctionTransformer

rg = np.random.RandomState(seed=42)
selector = FunctionTransformer(lambda x: select_random_features(x, sample_size=0.9, random_state=rg), accept_sparse=True)

def random_features_clfs(clf, n_estimators= 5, sample_size=0.9, random_state= None):
	if random_state is None:
		random_state = 0
	clfs = []
	for i in range(n_estimators):
		clfs.append(
			Pipeline([('random_selector', FunctionTransformer(lambda x: select_random_features(x
									, sample_size=sample_size
									, random_state=random_state+i))),
                    ('clf', clf) ])
			)
	return clfs
