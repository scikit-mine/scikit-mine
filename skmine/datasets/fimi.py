"""
Base IO for all FIMI datasets
All datasets are available here : `http://fimi.uantwerpen.be/data/`
"""
import os

import pandas as pd

from . import urlopen
from ._base import get_data_home

BASE_URL = 'http://fimi.uantwerpen.be/data/'

def preprocess(transaction):
    s = bytes.decode(transaction[:-2])
    return s.split(' ')

def fetch_any(filename):
    """Base loader for all datasets from the FIMI repository
    Each unique transaction will be represented as a Python list in the resulting data

    Parameters
    ----------
    filename : str
        Name of the file to fetch

    Returns
    -------
    pd.Series
        Transactions from the requested dataset,
        as an in-memory pandas Series
    """
    # TODO : downloading gzip file should be better
    data_home = get_data_home()
    filepath = os.path.join(data_home, filename)
    if filename in os.listdir(data_home):  # already fetched
        s = pd.read_pickle(filepath)
        return s
    else:                                  # not fetched yet
        url = BASE_URL + filename
        resp = urlopen(url)
        it = (preprocess(transaction) for transaction in resp)
        s = pd.Series(it)
        s.to_pickle(filepath)
        return s

def fetch_chess():
    """Fetch and return the chess dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting data

    Returns
    -------
    pd.Series
        Transactions from the chess dataset, as an in-memory pandas Series.

    Examples
    --------
    Let's say you want to load the chess dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_chess
    >>> data = fetch_chess()
    >>> data.head()
    0    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...
    1    [1, 3, 5, 7, 9, 12, 13, 15, 17, 19, 21, 23, 25...
    2    [1, 3, 5, 7, 9, 12, 13, 16, 17, 19, 21, 23, 25...
    3    [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 21, 23, 25...
    4    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...
    dtype: object
    """
    return fetch_any('chess.dat')


def fetch_connect():
    """Fetch and return the connect dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting data

    Returns
    -------
    pd.Series
        Transactions from the connect dataset, as an in-memory pandas Series.

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_connect
    >>> data = fetch_connect()
    >>> data.head()
    0    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    1    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    2    [1, 4, 7, 10, 13, 16, 19, 23, 25, 28, 31, 34, ...
    3    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    4    [1, 5, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, ...
    dtype: object
    """
    return fetch_any('connect.dat')


def fetch_mushroom():
    """Fetch and return the mushroom dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting data

    Returns
    -------
    pd.Series
        Transactions from the mushroom dataset, either as an in-memory pandas Series.

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_mushroom
    >>> data = fetch_mushroom()
    >>> data.head()
    0    [1, 3, 9, 13, 23, 25, 34, 36, 38, 40, 52, 54, ...
    1    [2, 3, 9, 14, 23, 26, 34, 36, 39, 40, 52, 55, ...
    2    [2, 4, 9, 15, 23, 27, 34, 36, 39, 41, 52, 55, ...
    3    [1, 3, 10, 15, 23, 25, 34, 36, 38, 41, 52, 54,...
    4    [2, 3, 9, 16, 24, 28, 34, 37, 39, 40, 53, 54, ...
    dtype: object
    """
    return fetch_any('mushroom.dat')


def fetch_pumsb():
    """Fetch and return the pumsb dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting data

    Returns
    -------
    pd.Series
        Transactions from the pumsb dataset, either as an in-memory pandas Series

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_pumsb
    >>> data = fetch_pumsb()
    >>> data.head()
    0    [0, 14, 17, 60, 66, 75, 84, 125, 155, 161, 163...
    1    [1, 15, 54, 60, 66, 74, 84, 111, 155, 161, 167...
    2    [0, 14, 17, 60, 66, 73, 84, 124, 155, 161, 163...
    3    [1, 15, 17, 60, 66, 73, 84, 111, 155, 161, 165...
    4    [2, 15, 17, 57, 70, 74, 84, 111, 160, 161, 167...
    dtype: object
    """
    return fetch_any('pumsb.dat')

def fetch_pumsb_star():
    """Fetch and return the pumsb_star dataset (Frequent Itemset Mining)
    Each unique transaction will be represented as a Python list in the resulting data

    Returns
    -------
    pd.Series
        Transactions from the pumsb_star dataset, either as an in-memory pandas Series

    Examples
    --------
    Let's say you want to load this dataset to benchmark your own mining algorithm
    >>> from skmine.datasets import fetch_pumsb_star
    >>> data = fetch_pumsb_star()
    >>> data.head()
    0    [0, 14, 60, 66, 75, 84, 125, 155, 161, 163, 16...
    1    [1, 15, 54, 60, 66, 74, 84, 111, 155, 161, 167...
    2    [0, 14, 60, 66, 73, 84, 124, 155, 161, 163, 16...
    3    [1, 15, 60, 66, 73, 84, 111, 155, 161, 165, 16...
    4    [2, 15, 57, 70, 74, 84, 111, 160, 161, 167, 16...
    dtype: object
    """
    return fetch_any('pumsb_star.dat', return_generator=return_generator)

# TODO kosarak, retail, accidents
