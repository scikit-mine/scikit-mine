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

    Examples
    --------
    Let's say you want to load the chess dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_chess
    >>> D = fetch_chess()
    >>> D.head()
    0    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...
    1    [1, 3, 5, 7, 9, 12, 13, 15, 17, 19, 21, 23, 25...
    2    [1, 3, 5, 7, 9, 12, 13, 16, 17, 19, 21, 23, 25...
    3    [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 21, 23, 25...
    4    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...
    name: chess, dtype: object
    """
    return fetch_any('chess.dat', data_home=data_home)


def fetch_connect(data_home=None):
    """Fetch and return the connect dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                       129
    Nb of transactions              67557
    Avg transaction size             43.0
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

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_connect
    >>> D = fetch_connect()
    >>> D.head()
    0    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    1    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    2    [1, 4, 7, 10, 13, 16, 19, 23, 25, 28, 31, 34, ...
    3    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    4    [1, 5, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    name: connect, dtype: object
    """
    return fetch_any('connect.dat', data_home=data_home)


def fetch_mushroom(data_home=None):
    """Fetch and return the mushroom dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                       119
    Nb of transactions               8124
    Avg transaction size             23.0
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

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_mushroom
    >>> D = fetch_mushroom()
    >>> D.head()
    0    [1, 3, 9, 13, 23, 25, 34, 36, 38, 40, 52, 54, ...
    1    [2, 3, 9, 14, 23, 26, 34, 36, 39, 40, 52, 55, ...
    2    [2, 4, 9, 15, 23, 27, 34, 36, 39, 41, 52, 55, ...
    3    [1, 3, 10, 15, 23, 25, 34, 36, 38, 41, 52, 54,...
    4    [2, 3, 9, 16, 24, 28, 34, 37, 39, 40, 53, 54, ...
    name: mushroom, dtype: object
    """
    return fetch_any('mushroom.dat', data_home=data_home)


def fetch_pumsb(data_home=None):
    """Fetch and return the pumsb dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                      2113
    Nb of transactions              49046
    Avg transaction size             74.0
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

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_pumsb
    >>> D = fetch_pumsb()
    >>> D.head()
    0    [0, 14, 17, 60, 66, 75, 84, 125, 155, 161, 163...
    1    [1, 15, 54, 60, 66, 74, 84, 111, 155, 161, 167...
    2    [0, 14, 17, 60, 66, 73, 84, 124, 155, 161, 163...
    3    [1, 15, 17, 60, 66, 73, 84, 111, 155, 161, 165...
    4    [2, 15, 17, 57, 70, 74, 84, 111, 160, 161, 167...
    name: pumsb, dtype: object
    """
    return fetch_any('pumsb.dat', data_home=data_home)


def fetch_pumsb_star(data_home=None):
    """Fetch and return the pumsb_star dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                      2088
    Nb of transactions              49046
    Avg transaction size            50.48
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

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_pumsb_star
    >>> D = fetch_pumsb_star()
    >>> D.head()
    0    [0, 14, 60, 66, 75, 84, 125, 155, 161, 163, 16...
    1    [1, 15, 54, 60, 66, 74, 84, 111, 155, 161, 167...
    2    [0, 14, 60, 66, 73, 84, 124, 155, 161, 163, 16...
    3    [1, 15, 60, 66, 73, 84, 111, 155, 161, 165, 16...
    4    [2, 15, 57, 70, 74, 84, 111, 160, 161, 167, 16...
    name: pumsb_star, dtype: object
    """
    return fetch_any('pumsb_star.dat', data_home=data_home)


def fetch_kosarak(data_home=None):
    """Fetch and return the kosarak dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    ====================   ==============
    Nb of items                     36855
    Nb of transactions             990002
    Avg transaction size              8.1
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
    ====================   ==============

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

def fetch_accidents(data_home=None):
    """Fetch and return the accidents dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    see: http://fimi.uantwerpen.be/data/accidents.pdf

    ====================   ==============
    Nb of items                     16470
    Nb of transactions              88162
    Avg transaction size             10.3
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `~/scikit_mine_data` subfolders.

    Returns
    -------
    pd.Series
        Transactions from the accidents dataset, as an in-memory pandas Series

    """
    return fetch_any('accidents.dat', data_home=data_home)
