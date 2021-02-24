"""
Base IO for all FIMI datasets
All datasets are available here : `http://fimi.uantwerpen.be/data/`
"""
import os

import pandas as pd

from .conf import urlopen
from ._base import get_data_home

BASE_URL = "http://fimi.uantwerpen.be/data/"


def _preprocess(transaction):
    s = bytes.decode(transaction[:-2])
    return s.split(" ")


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
    name, _ = os.path.splitext(filename)
    if filename in os.listdir(data_home):  # already fetched
        s = pd.read_pickle(filepath)
        s.name = name
    else:  # not fetched yet
        url = BASE_URL + filename
        resp = urlopen(url)
        it = (_preprocess(transaction) for transaction in resp)
        s = pd.Series(it, name=name)
        s.to_pickle(filepath)

    try:
        s = s.map(lambda l: list(map(int, l)))
    except ValueError:
        print(f"Could not cast {filename} items to integers")

    return s


def fetch_chess(data_home=None):
    """Fetch and return the chess dataset (Frequent Itemset Mining)

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
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("chess.dat", data_home=data_home)


def fetch_connect(data_home=None):
    """Fetch and return the connect dataset (Frequent Itemset Mining).

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
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("connect.dat", data_home=data_home)


def fetch_mushroom(data_home=None, return_D_y=False):
    """Fetch and return the mushroom dataset (Frequent Itemset Mining)

    The Mushroom data set includes descriptions of hypothetical samples corresponding
    to 23 species of gilled mushrooms in the Agaricus and Lepiota Family.

    It contains information about 8124 mushrooms (transactions).
    4208 (51.8%) are edible and 3916 (48.2%) are poisonous.

    The data contains 22 nomoinal features plus the class attribure (edible or not).
    These features were translated into 117 items.

    ====================   ==============
    Nb of items                       117
    Nb of transactions               8124
    Avg transaction size             22.0
    Density                         0.188
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    return_D_y: bool, default=False.
        If True, returns a tuple for both the data and the associated labels
        (0 for edible, 1 for poisonous)

    Returns
    -------
    D: pd.Series
        Transactions from the mushroom dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    (D, y) : tuple
        if ``return_D_y`` is True

    Examples
    --------
    >>> from skmine.datasets.fimi import fetch_mushroom
    >>> from skmine.datasets.utils import describe
    >>> D, y = fetch_mushroom(return_D_y=True)
    >>> describe(D)['n_items']
    117
    >>> y.value_counts()
    0    4208
    1    3916
    Name: mushroom, dtype: int64
    """
    mush = fetch_any("mushroom.dat", data_home=data_home)
    D = mush.str[1:]
    if return_D_y:
        y = mush.str[0]
        y = y.replace(2, 0)  # 2 is edible, 1 is poisonous
        return D, y

    return D


def fetch_pumsb(data_home=None):
    """Fetch and return the pumsb dataset (Frequent Itemset Mining)

    The Pumsb dataset contains census data for population and housing.

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
        Transactions from the pumsb dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("pumsb.dat", data_home=data_home)


def fetch_pumsb_star(data_home=None):
    """Fetch and return the pumsb_star dataset (Frequent Itemset Mining)

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
        Transactions from the pumsb_star dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("pumsb_star.dat", data_home=data_home)


def fetch_kosarak(data_home=None):
    """Fetch and return the kosarak dataset (Frequent Itemset Mining)

    Click-stream data from a hungarian on-line news portal.

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
        Transactions from the kosarak dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("kosarak.dat", data_home=data_home)


def fetch_retail(data_home=None):
    """Fetch and return the retail dataset (Frequent Itemset Mining)

    Contains market basket data from a Belgian retail store, anonymized.

    see: http://fimi.uantwerpen.be/data/retail.pdf

    ====================   ==============
    Nb of items                     16470
    Nb of transactions              88162
    Avg transaction size             10.3
    Densisty                     0.000626
    ====================   ==============

    Retail market basket data set supplied by a anonymous Belgian retail supermarket store.

    Results in approximately 5 months of data.
    The total amount of receipts being collected equals 88,163.

    In total, 5,133 customers have purchased at least one product during the data collection period

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the retail dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("retail.dat", data_home=data_home)


def fetch_accidents(data_home=None):
    """Fetch and return the accidents dataset (Frequent Itemset Mining)

    Traffic accident data, anonymized.

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
        Transactions from the accidents dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    """
    return fetch_any("accidents.dat", data_home=data_home)
