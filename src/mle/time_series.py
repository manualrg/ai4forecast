import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_targets(y, tau):
    return np.log(y.shift(-tau).div(y)).iloc[:-tau]

def run_adf_test(df: pd.DataFrame, *args, **kwargs):

    select_cols = ['adf', 'pvalue']
    res = df.apply(lambda col: adfuller(col), *args, **kwargs, axis=0)
    res.index = ['adf', 'pvalue', 'usedlag', 'nobs', 'crit_values', 'icbest']

    return res.T[select_cols]

def get_wave_features(X, periods=[12], n_harmonics=1):
    X_wave = pd.DataFrame(dtype=float)
    for tau in periods:
        for i in range(1, n_harmonics+1):
            X_wave[f"comp_{tau}_sin{i}"] = np.sin((X - 1) * (2 * np.pi / tau)) ** i
            X_wave[f"comp_{tau}_cos{i}"] = np.cos((X - 1) * (2 * np.pi / tau)) ** i

    return X_wave

def compute_macd(x: pd.DataFrame, short_tau: int, long_tau: int):
    short = x.rolling(short_tau, min_periods=1).mean()
    long = x.rolling(long_tau, min_periods=1).mean()
    return short.subtract(long)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MACD(BaseEstimator, TransformerMixin):
    """
    A wrapper to fit several LinearRegression (Base model) models to different sets of features and targets.
    Each model shares the same set of parameters (Base model parameters)
    For each column in targets, a Base model is fit. A mapping of feature columns for each model should be
    supplied, otherwise, regular multioutput regression is fit (each Base model shares features)
    Parameters
    ----------
    xcols = list,  each element is a list of column indexes used as features for each model
    Attributes
    ----------
    models_ = list, each element is a fitted instance of Base mode, on its respective features-target
    n_models_ = int, number of models to be fit. Inferred from ycols length
    n_features_in_ = float, average number of features used across each model
    coef_ = list, each element is an array with coefficients values, each array length may vary
    intercept_ = list, ach element is an intercept
    See Also
    --------
    sklearn.linear_model.LinearRegression
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/linear_model/_base.py#L389
    """

    def __init__(self, short_tau=5, long_tau=20):
        self.short_tau = short_tau
        self.long_tau = long_tau

    def fit(self, X=None, y=None, **fit_params):
        """
        Fit each model on its correspondent set of features-targets. Retrieves column metadata in xcols
        :param X: array-like, full set of features
        :param y: array-like, full set of targets
        :return: returns an instance of self.
        """

        return self

    def transform(self, X, **transform_params):
        """
        Get predictions for each model based on its correspondent set of features. Retrieves column metadata in xcols
        Predictions are horizontally stacked.
        :param X: array-like, Full set of features
        :return: array-like, 2d array of predictions
        """
        X = check_array(X)

        short = pd.DataFrame(X).rolling(self.short_tau, min_periods=1).mean()
        long = pd.DataFrame(X).rolling(self.long_tau, min_periods=1).mean()

        return short-long