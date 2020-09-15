"""
Base IO for the Instacart dataset
The dataset is available here : `https://www.instacart.com/datasets/grocery-shopping-2017`
"""
import os
import tarfile
import pandas as pd
import numpy as np

try:
    from lxml import html

    LXML_INSTALLED = True
except ImportError:
    LXML_INSTALLED = False

from .conf import urlopen
from ._base import get_data_home

_IMPORT_MSG = """
lxml is required to install the instacart dataset.
Please run `pip install lxml` before using instacart.
"""


def fetch_instacart(data_home=None):
    """Fetch/load function for the instacart dataset

    see: https://www.instacart.com/datasets/grocery-shopping-2017

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-mine data is stored in `~/scikit_mine_data/` subfolders.

    References
    ----------
    .. [1] “The Instacart Online Grocery Shopping Dataset 2017”
            Accessed from https://www.instacart.com/datasets/grocery-shopping-2017

    Notes
    -----
    This returns instacart transactions as a pd.Series, note that you still have access to all
    other data downloaded in your ``data_home`` path

    As instacart is a big dataset, ``fetch_instacart`` is designed to make the less computation
    as possible. If you already have downloaded the instacart dataset, just place
    `instacart.tar.gz` in your ``data_home``, or pass a ``data_home`` argument
    and ``fetch_instacart`` will start from there.

    Raises
    ------
    ImportError
        In case download is necessary and the ``lxml`` package is not installed.

    See Also
    --------
    get_data_home

    Examples
    --------
        >>> from skmine.datasets import fetch_instacart
        >>> D = fetch_instacart(data_home='~/scikit_mine_data/')  # doctest: +SKIP

    Returns
    -------
    pd.Series
        Customers orders. Each unique transaction will be represented as a Python list
    """
    data_home = data_home or get_data_home()
    data_home = os.path.expanduser(data_home)
    tar_filename = _download(data_home)
    data_path = os.path.join(data_home, "instacart_2017_05_01")

    if not os.path.exists(data_path):  # tarfile present but not uncompressed
        tar = tarfile.open(tar_filename, "r:gz")
        tar.extractall(data_home)
        tar.close()

    final_path = os.path.join(data_path, "transactions.pkl")
    if os.path.exists(final_path):
        s = pd.read_pickle(final_path)
    else:
        orders = _get_orders(data_path)
        s = orders.groupby("order_id")["product_name"].apply(np.unique)
        s.to_pickle(final_path)
    return s


def _get_orders(data_path):
    orders_path = os.path.join(data_path, "orders_postprocessed.pkl")
    if os.path.exists(orders_path):
        return pd.read_pickle(orders_path)
    order_products_path = os.path.join(data_path, "order_products__prior.csv")
    products_path = os.path.join(data_path, "products.csv")
    orders = pd.read_csv(order_products_path, usecols=["order_id", "product_id"])
    products = pd.read_csv(
        products_path, usecols=["product_id", "product_name", "aisle_id"]
    )
    orders = orders.merge(products, on="product_id", how="inner")
    orders.to_pickle(orders_path)
    return orders


def _download(data_home):
    tar_filename = os.path.join(data_home, "instacart.tar.gz")

    if os.path.exists(tar_filename):
        print("found instacart dataset at {}, fetching from there".format(tar_filename))
    elif not LXML_INSTALLED:
        raise ImportError(_IMPORT_MSG)
    else:
        print("downloading instacart dataset, this may take a while")
        data_link = "https://www.instacart.com/datasets/grocery-shopping-2017"
        tree = html.fromstring(urlopen(data_link).read())
        buttons = tree.xpath("//*[contains(@class, 'ic-btn ic-btn-success ic-btn-lg')]")
        download_link = buttons[0].attrib["href"]
        instacart_filedata = urlopen(download_link)
        targz_data = instacart_filedata.read()
        with open(tar_filename, "wb") as f:
            f.write(targz_data)
    return tar_filename
