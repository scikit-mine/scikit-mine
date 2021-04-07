"""
Base IO for all periodic datasets
"""
import os
from zipfile import ZipFile

import pandas as pd

from ._base import get_data_home
from .conf import urlopen


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


def _extract_canadian_tv(zip_name, filename, data_home):
    resp = urlopen(
        "https://applications.crtc.gc.ca/OpenData/Television%20Logs/STAR2/2020/2020-08.zip"
    )
    with open(zip_name, "wb") as fd:
        fd.write(resp.read())
    zf = ZipFile(zip_name)
    zf.extract(filename, path=data_home)


def fetch_canadian_tv(data_home=None, filename="2020-08\CBC_202008_140114251.log"):
    """
    Fetch and return canadian TV logs from August 8, 2020

    see: https://open.canada.ca/data/en/dataset/800106c1-0b08-401e-8be2-ac45d62e662e

    If the dataset has never been downloaded before, it will be downloaded and stored.

    The returned dataset contains only TV series episodes index by their associated timestamps
    Adverts are ignored when loading the dataset.

    ==============================      =======================================
    Number of events                    131
    Average delta per event             Timedelta('15 days 14:59:42.137404580')
    Average nb of points per event      15.977099236641221
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

    Examples
    --------
    >>> from skmine.datasets import fetch_canadian_tv
    >>> ctv = fetch_canadian_tv()  # first time will take long
    >>> ctv.head()
    2020-08-01 06:00:00            The Moblees
    2020-08-01 06:11:00    Big Block Sing Song
    2020-08-01 06:13:00    Big Block Sing Song
    2020-08-01 06:15:00               CBC Kids
    2020-08-01 06:15:00               CBC Kids
    dtype: object
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, squeeze=True, dtype="string", sep="\t")
    zip_name = filename.replace(".csv", ".zip")

    if not filename in os.listdir(data_home):
        _extract_canadian_tv(zip_name, filename, data_home)

    s = pd.read_csv(p, **kwargs)
    s = s.str.split("   ").str[1]  # TODO this is dirty. Store file on zenodo
    s = s[s.str.match("\d+[^\d]+")]
    index = s.str[:10]
    data = s.str[24:]
    s = pd.Series(data.values, index=index.values)
    s.index = pd.to_datetime(s.index, format="%y%m%d%H%M")
    return s
