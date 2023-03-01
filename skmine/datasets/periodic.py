"""
Base IO for all periodic datasets
"""
import os
import warnings
import pandas as pd

from ._base import get_data_home


def fetch_file(filepath, separator=','):
    """Loader for files in periodic format (timestamp,event\n). The first element can be a datetime or an integer and
    the second is a string.
    This file reader can also work for files with only one value per line (the event).
    The indexes then correspond to the line numbers.

    Parameters
    ----------
    filepath : str
        Path of the file to load

    separator : str
        Indicate a custom separator between timestamps and events. By default, it is a comma.
        If the file contains only one column, this parameter is not useful.

    Returns
    -------
    pd.Series
        Logs from the custom dataset, as an in-memory pandas Series.
        Events are indexed by timestamps.
    """
    s = pd.read_csv(filepath, sep=separator, header=None, dtype="string").squeeze(axis="columns")
    if type(s) == pd.DataFrame:
        s = pd.Series(s[1].values, index=s[0])
        try:
            s.index = pd.to_datetime(s.index)
        except ValueError:
            s.index = s.index.astype("int64")
    s.index.name = "timestamp"
    s.name = filepath
    return s


def fetch_health_app(data_home=None, filename="health_app.csv"):
    """Fetch and return the health app log dataset

    see: https://github.com/logpai/loghub

    HealthApp is a mobile application for Android devices.
    Logs were collected from an Android smartphone after 10+ days of use.

    Logs have been grouped by their types, hence resulting
    in only 20 different events.

    ==============================      ===================================
    Number of occurrences               2000
    Number of events                    20
    Average delta per event             Timedelta('0 days 00:53:24.984000')
    Average nb of points per event      100.0
    ==============================      ===================================

    Parameters
    ----------
    filename : str, default: health_app.csv
        Name of the file (without the data_home directory) where the dataset will be or is already downloaded.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        System logs from the health app dataset, as an in-memory pandas Series.
        Events are indexed by timestamps.
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, index_col=0, dtype="string")
    if filename in os.listdir(data_home):
        s = pd.read_csv(p, **kwargs).squeeze(axis="columns")
    else:
        s = pd.read_csv(
            "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log",
            sep="|",
            on_bad_lines='skip',
            usecols=[0, 1],
            **kwargs,
        ).squeeze(axis="columns")
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
    Number of occurrences               2093
    Number of events                    98
    Average delta per event             Timedelta('19 days 02:13:36.122448979')
    Average nb of points per event      21.35714285714285
    ==============================      =======================================

    Parameters
    ----------
    filename : str, default: canadian_tv.txt
        Name of the file (without the data_home directory) where the dataset will be or is already downloaded.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        TV series events from canadian TV, as an in-memory pandas Series.
        Events are indexed by timestamps.

    Notes
    -----
    For now the entire .zip file is downloaded, being ~90mb on disk
    Downloading preprocessed dataset from zenodo.org is something we consider.

    See Also
    -------
    skmine.datasets.get_data_home
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, dtype="string", index_col=0)

    if filename not in os.listdir(data_home):
        s = pd.read_csv(
            "https://zenodo.org/record/4671512/files/canadian_tv.txt", **kwargs).squeeze(axis="columns")
        s.to_csv(p, index=True, header=False)
    else:
        s = pd.read_csv(p, **kwargs).squeeze(axis="columns")

    s.index = pd.to_datetime(s.index)
    s.index.name = "timestamp"
    s.name = "canadian_tv"
    return s
