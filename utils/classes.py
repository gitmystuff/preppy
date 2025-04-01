import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from . import functions as utils
from sklearn.model_selection import train_test_split

class TrainTestSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.2, random_state=None, stratify=None):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, X, y=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify
        )
        return self

    def transform(self, X, y=None):
        # The transform method is not really used in this case,
        # but we need to return something.
        return X

    def get_splits(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

class ConstantAndSemiConstantRemover(BaseEstimator, TransformerMixin):
    """
    A transformer that removes constant and semi-constant (99% threshold) columns.

    Parameters:
    -----------
    threshold : float, optional
        The threshold for semi-constant columns. Defaults to 0.99 (99%).

    Attributes:
    -----------
    non_constant_columns_ : list
        The indices of the non-constant and non-semi-constant columns.
    """

    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.non_constant_columns_ = None

    def fit(self, X, y=None):
        """
        Fits the transformer to the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            The input data.
        y : array-like of shape (n_samples,), optional
            The target values. Ignored.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        n_samples = X_values.shape[0]
        self.non_constant_columns_ = []

        for i in range(X_values.shape[1]):
            unique_values = np.unique(X_values[:, i])
            if len(unique_values) > 1:  # Not constant
                for val in unique_values:
                    if np.sum(X_values[:, i] == val) / n_samples <= self.threshold:
                        self.non_constant_columns_.append(i)
                        break # column is not semi constant, go to next column.

        self.non_constant_columns_ = sorted(list(set(self.non_constant_columns_))) #remove duplicates and sort
        return self

    def transform(self, X):
        """
        Transforms the input data by removing constant and semi-constant columns.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            The input data.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features_transformed) or pandas DataFrame
            The transformed data.
        """
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.non_constant_columns_]
        else:
            return X[:, self.non_constant_columns_]

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits to data, then transforms it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Input samples.
        y : array-like of shape (n_samples,), optional
            Target values (ignored).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new) or pandas DataFrame.
            Transformed array.
        """
        self.fit(X, y)
        return self.transform(X)

class DuplicateRemoverDF(BaseEstimator, TransformerMixin):
    """
    A transformer that removes duplicate rows and columns from a Pandas DataFrame.

    Parameters:
    -----------
    remove_rows : bool, optional
        Whether to remove duplicate rows. Defaults to True.
    remove_columns : bool, optional
        Whether to remove duplicate columns. Defaults to True.

    Attributes:
    -----------
    original_columns_ : list
        The original column indices before removing duplicates.
    original_rows_ : list
        The original row indices before removing duplicates.

    """

    def __init__(self, remove_rows=True, remove_columns=True):
        self.remove_rows = remove_rows
        self.remove_columns = remove_columns
        self.original_columns_ = None
        self.original_rows_ = None

    def fit(self, X, y=None):
        """
        Fits the transformer to the input data.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), optional
            The target values. Ignored.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        if self.remove_columns:
            _, self.original_columns_ = np.unique(X.values, axis=1, return_index=True)
        if self.remove_rows:
            _, self.original_rows_ = np.unique(X.values, axis=0, return_index=True)
        return self

    def transform(self, X):
        """
        Transforms the input data by removing duplicate rows and columns.

        Parameters:
        -----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_transformed : pandas DataFrame of shape (n_samples_transformed, n_features_transformed)
            The transformed data.
        """
        if self.remove_columns:
            X = X.iloc[:, self.original_columns_]
        if self.remove_rows:
            X = X.iloc[self.original_rows_, :]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits to data, then transforms it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            Target values (ignored).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : pandas DataFrame of shape (n_samples_new, n_features_new)
            Transformed array.
        """
        self.fit(X, y)
        return self.transform(X)
