import os
import os.path
import pandas as pd
import tarfile

import pytest

from .. import _instacart
from .._instacart import fetch_instacart
from .._instacart import _get_orders

def mock_urlopen(url):
    for i in range(2):
        transaction = ' '.join('{}'.format(i*j) for j in range(2)) + ' \n'
        yield bytes(transaction, encoding='utf-8')

def mock_read_pickle(*args, **kwargs):
    return pd.Series([[1, 2, 3], [4, 5]])


def mock_read_csv_orders(*args, **kwargs):
    usecols = kwargs.pop('usecols')
    product_ids, d = [1, 2], dict()
    if 'order_id' in usecols:
        d = dict(order_id=[10, 30], product_id=product_ids)
    if 'product_name' in usecols:
        d = dict(product_name=['Butter', 'Tooth Paste'], product_id=product_ids)
    return pd.DataFrame(d)


def mock_get_orders(*args, **kwargs):
    d = dict(
        order_id=[10, 30, 40],
        product_id=[1, 3, 4],
        product_name=['eggs', 'milk', 'bananas'],  # aisle_id not required for now
    )
    return pd.DataFrame(d)


def test_download(monkeypatch):
    monkeypatch.setattr(_instacart, 'LXML_INSTALLED', False)
    with pytest.raises(ImportError):
        _instacart._download('fake_data_home')

def test_get_orders_already_fetched(monkeypatch):
    monkeypatch.setattr(os.path, 'exists', lambda *args : True)  # enforce branching
    monkeypatch.setattr(pd, 'read_pickle', mock_read_pickle)
    orders = _get_orders('random_path')
    assert isinstance(orders, pd.Series)
    assert orders.shape == (2,)

def test_get_orders_not_fetched(monkeypatch):
    monkeypatch.setattr(os.path, 'exists', lambda *args : False)  # enforce branching
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_orders)
    monkeypatch.setattr(pd.DataFrame, 'to_pickle', lambda *args: None)
    orders = _get_orders('random_path')
    assert isinstance(orders, pd.DataFrame)
    assert 'product_id' in orders.columns
    assert 'product_name' in orders.columns
    assert 'order_id' in orders.columns

def test_fetch_orders_already_fetched(monkeypatch):
    monkeypatch.setattr(os.path, 'exists', lambda *ars: True)  # enforce branching
    monkeypatch.setattr(pd, 'read_pickle', mock_read_pickle)
    data = fetch_instacart('fake_data_home')
    assert data.shape == (2, )
 

def test_fetch_orders_not_fetched(monkeypatch):
    monkeypatch.setattr(_instacart, '_download', lambda *args: 'fake_file.tar.gz')
    monkeypatch.setattr(os.path, 'exists', lambda path: '.pkl' not in path)  # enforce branching
    monkeypatch.setattr(_instacart, '_get_orders', mock_get_orders)
    monkeypatch.setattr(pd.Series, 'to_pickle', lambda *args: None)
    data = fetch_instacart('fake_data_home')
    assert isinstance(data, pd.Series)
    assert data.shape == (3, )