"""
Base IO for all periodic datasets
"""
import os

import pandas as pd

from ._base import get_data_home


def fetch_health_app(data_home=None, filename="health_app.csv"):
    """Fetch and return the health app log dataset
    from github.com/logpai/loghub

    HealthApp is a mobile application for Android devices.
    Logs were collected from an Android smartphone after 10+ days of use.

    This dataset only represents the different types of logs, hence resulting
    in only 20 different events.

    ==============================      ==================================
    Number of events                    20
    Average delta per event             Timedelta('0 days 00:53:24.984000')
    Average nb of points per event      100.0
    ========================            ===================================

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the health app dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, index_col=0, squeeze=True, dtype="string")
    if filename in os.listdir(data_home):
        s = pd.read_csv(p, index_col=0, squeeze=True)
    else:
        s = pd.read_csv(
            "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log",
            sep="|",
            error_bad_lines=False,
            usecols=[0, 1],
            **kwargs
        )
        s.to_csv(p)
    s.index.name = "timestamp"
    s.index = pd.to_datetime(s.index, format="%Y%m%d-%H:%M:%S:%f")

    return s
