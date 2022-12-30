"""
Base IO for all FIMI datasets
All datasets are available here : `http://fimi.uantwerpen.be/data/`
"""
import os
import wget
import gzip

import pandas as pd

from ._base import get_data_home

BASE_URL_FIMI = "http://fimi.uantwerpen.be/data/"
BASE_URL_CGI = "https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/"


def _read_dat(filepath, int_values=True, separator=' ', zip=False):
    """Read a local dataset file whose separator can be customized and whose values are either integers or strings.

    Parameters
    ----------
    filepath : str
        Indicate the path of the file to be read

    int_values : bool
        Specify if the items in the file are all integers. If not, then the items are considered as strings.

    separator : str
        Specify a separator between items other than the default space

    Returns
    -------
    list
        Return a transaction list composed for each transaction of a list of items
    """
    try:
        if zip:
            with gzip.open(filepath, 'rt') as f:
                lines = f.read().splitlines()
        else:
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
    except UnicodeDecodeError:
        print(f"The file {filepath} is already present in your data_home but it is a binary file, it must be "
              f"deleted.")
        raise

    transactions = [[int(item) if int_values else item for item in line.rstrip().split(separator)] for line in lines]

    return transactions


def fetch_file(filepath, separator=' ', int_values=False):
    """Loader for files in FIMI format

    Parameters
    ----------
    filepath : str
        Path of the file to load

    separator : str
        Indicate a custom separator between the items. By default, it is a space.

    int_values : bool, default=False
        Specify if the items in the file are all integers. If not, then the items are considered as strings.
        With integers, the algorithms are more efficient.

    Returns
    -------
    pd.Series
        Transactions from the requested dataset,
        as an in-memory pandas Series
    """
    transactions = _read_dat(filepath, int_values=int_values, separator=separator)
    s = pd.Series(transactions, name=os.path.splitext(os.path.basename(filepath))[0])

    return s


def fetch_any(filename, base_url=BASE_URL_FIMI, data_home=None):
    """Base loader for all datasets from the FIMI and CGI repository
    Each unique transaction will be represented as a Python list in the resulting pandas Series

    see: http://fimi.uantwerpen.be/data/
    https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/dataSets.html

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `~/scikit_mine_data/` subfolders.

    filename : str
        Name of the file to fetch

    base_url : str
        URL indicating where to fetch the dataset

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
        it = _read_dat(filepath, zip=True) if base_url == BASE_URL_CGI else _read_dat(filepath)
    else:  # not fetched yet
        url = base_url + filename
        wget.download(url, filepath)
        it = _read_dat(filepath, zip=True) if base_url == BASE_URL_CGI else _read_dat(filepath)

    s = pd.Series(it, name=name)
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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the chess dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("chess.dat", base_url=BASE_URL_FIMI, data_home=data_home)


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the connect dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("connect.dat", base_url=BASE_URL_FIMI, data_home=data_home)


def fetch_mushroom(data_home=None, return_y=False):
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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    return_y: bool, default=False.
        If True, returns a tuple for both the data and the associated labels
        (0 for edible, 1 for poisonous)

    Returns
    -------
    mush : pd.Series
        Transactions from the mushroom dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    (mush, y) : tuple
        if ``return_y`` is True

    Examples
    --------
    >>> from skmine.datasets.fimi import fetch_mushroom
    >>> from skmine.datasets.utils import describe
    >>> X, y = fetch_mushroom(return_y=True)
    >>> describe(X)['n_items']
    119
    >>> y.value_counts()
    0    4208
    1    3916
    Name: mushroom, dtype: int64
    """
    mush = fetch_any("mushroom.dat", base_url=BASE_URL_FIMI, data_home=data_home)
    if return_y:
        y = mush.str[0].replace(2, 0)  # 2 is edible, 1 is poisonous
        X = mush.str[1:]
        return mush, y

    return mush


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the pumsb dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("pumsb.dat", base_url=BASE_URL_FIMI, data_home=data_home)


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the pumsb_star dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("pumsb_star.dat", base_url=BASE_URL_FIMI, data_home=data_home)


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the kosarak dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("kosarak.dat", base_url=BASE_URL_FIMI, data_home=data_home)


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the retail dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.
    """
    return fetch_any("retail.dat", base_url=BASE_URL_FIMI, data_home=data_home)


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
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `~/scikit_mine_data`.

    Returns
    -------
    pd.Series
        Transactions from the accidents dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    """
    return fetch_any("accidents.dat", base_url=BASE_URL_FIMI, data_home=data_home)


def fetch_iris(data_home=None, return_y=False):
    """Fetch and return the discretized iris dataset (Frequent Itemset Mining)

    This dataset corresponds to the iris dataset which has been discretized into 19 items.
    The last column (items: 17, 18, 19) corresponds to the targets and can be useful for classification.

    see: https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/exmpleDNnotes.html#iris

    ====================   ==============
    Nb of items                        19
    Nb of transactions                150
    Avg transaction size                5
    Density                         26.32
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `~/scikit_mine_data`.

    return_y : bool, default: False
        If True, returns a tuple for both the data and the associated labels.

    Returns
    -------
    pd.Series
        Transactions from the iris dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    """
    iris = fetch_any("iris.D19.N150.C3.num.gz", base_url=BASE_URL_CGI, data_home=data_home)
    if return_y:
        y = iris.str[-1]
        X = iris.str[:-1]
        return X, y
    return iris


def fetch_breast(data_home=None, return_y=False):
    """Fetch and return the discretized breast dataset (Frequent Itemset Mining)

    This dataset corresponds to the breast dataset which has been discretized into 20 items.
    The last column (items: 19, 20) corresponds to the targets and can be useful for classification.

    see: https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/exmpleDNnotes.html#breast

    ====================   ==============
    Nb of items                        20
    Nb of transactions                699
    Avg transaction size             9.98
    Density                            50
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `~/scikit_mine_data`.

    return_y : bool, default: False
        If True, returns a tuple for both the data and the associated labels.

    Returns
    -------
    pd.Series
        Transactions from the breast dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    """
    breast = fetch_any("breast.D20.N699.C2.num", base_url=BASE_URL_CGI, data_home=data_home)
    if return_y:
        y = breast.str[-1]
        X = breast.str[:-1]
        return X, y
    return breast


def fetch_tictactoe(data_home=None, return_y=False):
    """Fetch and return the discretized tictactoe dataset (Frequent Itemset Mining)

    This dataset corresponds to the tictactoe dataset which has been discretized into 29 items.
    The last column (items: 28, 29) corresponds to the targets and can be useful for classification.

    see: https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/exmpleDNnotes.html#tictactoe

    ====================   ==============
    Nb of items                        29
    Nb of transactions                958
    Nb of transactions                699
    Avg transaction size               10
    Density                         34.48
    ====================   ==============

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `~/scikit_mine_data`.

    return_y : bool, default: False
        If True, returns a tuple for both the data and the associated labels.

    Returns
    -------
    pd.Series
        Transactions from the breast dataset, as an in-memory pandas Series.
        Each unique transaction is represented as a Python list.

    """
    tictactoe = fetch_any("ticTacToe.D29.N958.C2.num", base_url=BASE_URL_CGI, data_home=data_home)
    if return_y:
        y = tictactoe.str[-1]
        X = tictactoe.str[:-1]
        return X, y
    return tictactoe
