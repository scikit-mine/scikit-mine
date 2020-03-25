import os
import types

import pandas as pd
import pytest

from skmine.datasets import fimi
from skmine.datasets import get_data_home

def mock_urlopen(url):
    for i in range(2):
        transaction = ' '.join('{}'.format(i*j) for j in range(2)) + ' \n'
        yield bytes(transaction, encoding='utf-8')

def mock_read_pickle(*args, **kwargs):
    return pd.Series([[1, 2, 3], [4, 5]])


def test_fetch_any(monkeypatch):
    # force file to be in DATA_HOME, so we don't need to fetch it
    monkeypatch.setattr(os, 'listdir', lambda *args: ['imaginary_data.dat'])
    monkeypatch.setattr(pd, 'read_pickle', mock_read_pickle)
    data = fimi.fetch_any('imaginary_data.dat')
    assert data.shape == (2,)
    pd.testing.assert_series_equal(pd.Series([3, 2]), data.map(len))

    # now DATA_HOME to be empty, file has to be fetched
    monkeypatch.setattr(os, 'listdir', lambda *args: list())
    monkeypatch.setattr(fimi, 'urlopen', mock_urlopen)
    data = fimi.fetch_any('other_imaginary_dataset.dat')
    assert data.shape == (2, )
    pd.testing.assert_series_equal(pd.Series([2, 2]), data.map(len))
