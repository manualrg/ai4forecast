import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

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