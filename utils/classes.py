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