import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MACD(BaseEstimator, TransformerMixin):

    def __init__(self, short_tau=5, long_tau=20,
                 zscore_tau=60, flg_zcore=True):
        self.short_tau = short_tau
        self.long_tau = long_tau
        self.zscore_tau = zscore_tau
        self.flg_zcore = flg_zcore
    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):

        if self.flg_zcore:
            return pass

        return pass
