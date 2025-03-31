import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from . import functions as utils

class RemoveConstants(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    vars = utils.identify_consts(X)
    X.drop(vars, axis=1, inplace=True)
    return X.values

class RemoveQuasiConstants(BaseEstimator, TransformerMixin):
  def __init__(self, thresh=0.95):
    self.thresh=thresh

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    vars = utils.identify_quasi_consts(X, thresh=self.thresh)
    X.drop(vars, axis=1, inplace=True)
    return X.values

class DropDuplicates(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X.drop_duplicates(inplace=True) # drop duplicate rows
    vars = utils.check_col_duplicates(X) 
    X.drop(vars, axis=1, inplace=True) # drop duplicate columns
    return X.values

class HandleMissingValues(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = utils.handle_missing_values(X)
    return X.values

class PerformOHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        cat_features = []
        for feat in X.select_dtypes(include=['object', 'category']):
            if len(X[feat].value_counts()) < 3:
                X[feat] = X[feat].map({X[feat].value_counts().index[0]: 0, X[feat].value_counts().index[1]: 1})
                X[feat] = X[feat].astype(int)
            elif 2 < len(X[feat].value_counts()) < 6:
                cat_features.append(feat)
            elif len(X[feat].value_counts()) > 5:
                freq = X.groupby(feat, observed=False).size() / len(X)
                X[feat] = X[feat].map(freq)

        if cat_features:
            ohe = OneHotEncoder(categories='auto', drop='first', sparse_output=False, handle_unknown='ignore')
            ohe_array = ohe.fit_transform(X[cat_features])
            ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(cat_features), index=X.index)
            X = X.join(ohe_df)
            X.drop(cat_features, axis=1, inplace=True)
          
        return X
