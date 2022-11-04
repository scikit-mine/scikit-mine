import os
import types

import pandas as pd

import pytest

from skmine.datasets import fimi
from skmine.datasets import get_data_home


def test_read_dat_fimi():
    transactions = fimi._read_dat('./skmine/datasets/tests/test_fimi_file_int.dat')
    print(type(transactions))
    assert len(transactions) == 4
    assert all(type(item) == int for transaction in transactions for item in transaction)
    assert transactions == [[1, 2, 3], [4, 5], [2], [8, 9]]


def test_read_dat_text():
    transactions = fimi.fetch_file('./skmine/datasets/tests/test_fimi_file_text.dat')
    assert len(transactions) == 4
    assert all(type(item) == str for transaction in transactions for item in transaction)
    pd.testing.assert_series_equal(pd.Series([['dog', 'kitten', 'cat'], ['fish', 'hamster'], ['kitten'], ['8', 'mouse']]
                                             , name='test_fimi_file_text'), transactions)


def test_read_dat_text_separator_comma():
    transactions = fimi.fetch_file('./skmine/datasets/tests/test_fimi_file_text_sep_comma.dat', ',')
    assert len(transactions) == 4
    assert all(type(item) == str for transaction in transactions for item in transaction)
    pd.testing.assert_series_equal(pd.Series([['dog', 'kitten', 'cat'], ['fish', 'hamster'], ['kitten'], ['8', 'mouse']]
                                             , name='test_fimi_file_text_sep_comma'), transactions)


def test_fetch_any_not_fetched():
    transactions = fimi.fetch_any('chess.dat', './skmine/datasets/tests/')
    assert os.path.isfile('./skmine/datasets/tests/chess.dat')
    os.remove('./skmine/datasets/tests/chess.dat')
    assert transactions.name == 'chess'
    assert len(transactions) == 3196
    assert all(type(item) == int for transaction in transactions for item in transaction)


def test_fetch_any_already_fetched():
    transactions = fimi.fetch_any('test_fimi_file_int.dat', './skmine/datasets/tests/')
    assert len(transactions) == 4
    assert all(type(item) == int for transaction in transactions for item in transaction)
    pd.testing.assert_series_equal(pd.Series([[1, 2, 3], [4, 5], [2], [8, 9]], name='test_fimi_file_int'), transactions)
