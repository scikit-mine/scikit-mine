"""
Base IO for all periodic datasets
"""
import os
import warnings

import pandas as pd

from ._base import get_data_home


def deduplicate(S):
    """deduplicate pd.Series by removing rows with same index and values"""
    dedup = S.groupby(S.index).first()
    diff = len(S) - len(dedup)
    if diff:
        warnings.warn(f"found {diff} duplicates in S, removing them")
        return dedup
    return S


def fetch_health_app(data_home=None, filename="health_app.csv"):
    """Fetch and return the health app log dataset

    see: https://github.com/logpai/loghub

    HealthApp is a mobile application for Android devices.
    Logs were collected from an Android smartphone after 10+ days of use.

    Logs have been grouped by their types, hence resulting
    in only 20 different events.

    ==============================      ===================================
    Number of events                    20
    Average delta per event             Timedelta('0 days 00:53:24.984000')
    Average nb of points per event      100.0
    ==============================      ===================================

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        System logs from the health app dataset, as an in-memory pandas Series.
        Events are indexed by timestamps.
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, index_col=0, squeeze=True, dtype="string")
    if filename in os.listdir(data_home):
        s = pd.read_csv(p, **kwargs)
    else:
        s = pd.read_csv(
            "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log",
            sep="|",
            error_bad_lines=False,
            usecols=[0, 1],
            **kwargs,
        )
        s.to_csv(p, header=False)
    s.index.name = "timestamp"
    s.index = pd.to_datetime(s.index, format="%Y%m%d-%H:%M:%S:%f")

    return s


def fetch_canadian_tv(data_home=None, filename="canadian_tv.txt"):
    """
    Fetch and return canadian TV logs from August 2020

    see: https://zenodo.org/record/4671512

    If the dataset has never been downloaded before, it will be downloaded and stored.

    The returned dataset contains only TV series programs indexed by their associated timestamps.
    Adverts are ignored when loading the dataset.

    ==============================      =======================================
    Number of events                    98
    Average delta per event             Timedelta('19 days 02:13:36.122448979')
    Average nb of points per event      21.35714285714285
    ==============================      =======================================

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        TV series events from canadian TV, as an in-memory pandas Series.
        Events are indexed by timestamps.

    Notes
    -----
    For now the entire .zip file is downloaded, being ~90mb on disk
    Downloading preprocessd dataset from zenodo.org is something we consider.

    See Also
    -------
    skmine.datasets.get_data_home
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, squeeze=True, dtype="string", index_col=0,)

    if filename not in os.listdir(data_home):
        s = pd.read_csv(
            "https://zenodo.org/record/4671512/files/canadian_tv.txt", **kwargs
        )
        s.to_csv(p, index=True, header=False)
    else:
        s = pd.read_csv(p, **kwargs)

    s.index = pd.to_datetime(s.index)
    s.index.name = "timestamp"
    s.name = "canadian_tv"
    return s
