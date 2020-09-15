import os
import types

import pandas as pd
import pytest

from skmine.datasets import fimi
from skmine.datasets import get_data_home


def mock_urlopen(url):
    for i in range(2):
        transaction = " ".join("{}".format(i * j) for j in range(2)) + " \n"
        yield bytes(transaction, encoding="utf-8")


def mock_read_pickle(*args, **kwargs):
    return pd.Series([[1, 2, 3], [4, 5]])


def test_fetch_any(monkeypatch):
    name = "imaginary_dataset"
    # force file to be in DATA_HOME, so we don't need to fetch it
    monkeypatch.setattr(os, "listdir", lambda *args: ["{}.dat".format(name)])
    monkeypatch.setattr(pd, "read_pickle", mock_read_pickle)
    data = fimi.fetch_any("{}.dat".format(name))
    assert data.shape == (2,)
    pd.testing.assert_series_equal(pd.Series([3, 2]), data.map(len))

    # now DATA_HOME to be empty, file has to be fetched
    monkeypatch.setattr(os, "listdir", lambda *args: list())
    monkeypatch.setattr(fimi, "urlopen", mock_urlopen)
    name = "other_imaginary_dataset"
    data = fimi.fetch_any("{}.dat".format(name))
    assert data.shape == (2,)
    pd.testing.assert_series_equal(pd.Series([2, 2], name=name), data.map(len))
