import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import spearmanr

def get_targets(y, tau):
    """
    (y[t+1]-y[t])/y[t+1] approximates to: np.log(y[t+1]) - np.log(y[t]) = np.log(y[t+1]/y[t])
    https://stats.stackexchange.com/questions/244199/why-is-it-that-natural-log-changes-are-percentage-changes-what-is-about-logs-th
    :param y:
    :param tau:
    :return:
    """
    agg = y.rolling(tau).median()
    return np.log(agg.shift(-tau).div(agg)).iloc[tau:-tau]

def run_adf_test(df: pd.DataFrame, *args, **kwargs):

    select_cols = ['adf', 'pvalue']
    res_idx = ['adf', 'pvalue', 'usedlag', 'nobs', 'crit_values', 'icbest']
    res = df.apply(lambda col: pd.Series(index=res_idx, data=adfuller(col)), axis=0)

    return res.T[select_cols]

def get_wave_features(X, periods=[12], n_harmonics=1):
    X_wave = pd.DataFrame(dtype=float)
    for tau in periods:
        for i in range(1, n_harmonics+1):
            X_wave[f"comp_{tau}_sin{i}"] = np.sin((X - 1) * (2 * np.pi / tau)) ** i
            X_wave[f"comp_{tau}_cos{i}"] = np.cos((X - 1) * (2 * np.pi / tau)) ** i

    return X_wave


def compute_spearman(y_true, y_pred):

    return spearmanr(y_true, y_pred, nan_policy='omit').correlation

