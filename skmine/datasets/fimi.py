"""
Base IO for all FIMI datasets
All datasets are available here : `http://fimi.uantwerpen.be/data/`
"""
import os

import pandas as pd

from . import urlopen
from ._base import get_data_home

BASE_URL = 'http://fimi.uantwerpen.be/data/'

def _preprocess(transaction):
    s = bytes.decode(transaction[:-2])
    return s.split(' ')

def fetch_any(filename, data_home=None):
    """Base loader for all datasets from the FIMI repository
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    see: http://fimi.uantwerpen.be/data/

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `~/scikit_mine_data/` subfolders.

    filename : str
        Name of the file to fetch

    Returns
    -------
    pd.Series
        Transactions from the requested dataset,
        as an in-memory pandas Series
    """
    data_home = data_home or get_data_home()
    filepath = os.path.join(data_home, filename)
    if filename in os.listdir(data_home):  # already fetched
        s = pd.read_pickle(filepath)
    else:                                  # not fetched yet
        url = BASE_URL + filename
        resp = urlopen(url)
        it = (_preprocess(transaction) for transaction in resp)
        name, _ = os.path.splitext(filename)
        s = pd.Series(it, name=name)
        s.to_pickle(filepath)

    return s

def fetch_chess(data_home=None):
    """Fetch and return the chess dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                        75
    Nb of transactions               3196
    Avg transaction size             37.0
    Density                         0.493
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the chess dataset, as an in-memory pandas Series.
    """
    return fetch_any('chess.dat', data_home=data_home)


def fetch_connect(data_home=None):
    """Fetch and return the connect dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                       129
    Nb of transactions              67557
    Avg transaction size             43.0
    Density                         0.333
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the connect dataset, as an in-memory pandas Series.
    """
    return fetch_any('connect.dat', data_home=data_home)


def fetch_mushroom(data_home=None):
    """Fetch and return the mushroom dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                       119
    Nb of transactions               8124
    Avg transaction size             23.0
    Density                         0.193
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the mushroom dataset, as an in-memory pandas Series.
    """
    return fetch_any('mushroom.dat', data_home=data_home)


def fetch_pumsb(data_home=None):
    """Fetch and return the pumsb dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                      2113
    Nb of transactions              49046
    Avg transaction size             74.0
    Density                         0.035
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the pumsb dataset, as an in-memory pandas Series
    """
    return fetch_any('pumsb.dat', data_home=data_home)


def fetch_pumsb_star(data_home=None):
    """Fetch and return the pumsb_star dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                      2088
    Nb of transactions              49046
    Avg transaction size            50.48
    Density                         0.024
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the pumsb_star dataset, as an in-memory pandas Series
    """
    return fetch_any('pumsb_star.dat', data_home=data_home)


def fetch_kosarak(data_home=None):
    """Fetch and return the kosarak dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                     36855
    Nb of transactions             990002
    Avg transaction size              8.1
    Density                      0.000220
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the kosarak dataset, as an in-memory pandas Series
    """
    return fetch_any('kosarak.dat', data_home=data_home)


def fetch_retail(data_home=None):
    """Fetch and return the retail dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    see: http://fimi.uantwerpen.be/data/retail.pdf

    ====================   ==============
    Nb of items                     16470
    Nb of transactions              88162
    Avg transaction size             10.3
    Densisty                     0.000626
    ====================   ==============

    Retail market basket data set supplied by a anonymous Belgian retail supermarket store.

    Results in approximately 5 months of data. The total amount of receipts being collected equals 88,163.

    In total, 5,133 customers have purchased at least one product in the supermarket during the data collection period

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the retail dataset, as an in-memory pandas Series
    """
    return fetch_any('retail.dat', data_home=data_home)

def fetch_accidents(data_home=None):
    """Fetch and return the accidents dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    see: http://fimi.uantwerpen.be/data/accidents.pdf

    ====================   ==============
    Nb of items                       468
    Nb of transactions             340183
    Avg transaction size           33.807
    Density                         0.072
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `~/scikit_mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the accidents dataset, as an in-memory pandas Series

    """
    return fetch_any('accidents.dat', data_home=data_home)
