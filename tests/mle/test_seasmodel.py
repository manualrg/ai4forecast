
from src.mle import time_series as mle_ts

from sklearn.utils.estimator_checks import check_estimators_pickle, check_estimator


def test_sklearn_comply():
    estimator = mle_ts.SeasOHE()
    for est, check in check_estimator(estimator, generate_only=True):
        try:
            check(est)
        except AssertionError as e:
            print('Failed: ', check, e)

test_sklearn_comply()