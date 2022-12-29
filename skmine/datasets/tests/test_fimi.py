import os
import types

import pandas as pd

import pytest
import requests

from skmine.datasets import fimi

def test_url_fimi():
    url_fimi = "http://fimi.uantwerpen.be/data/"
    resp = requests.get(url_fimi)
    assert resp.status_code == 200


def test_url_cgi():
    url_cgi = "https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/"
    resp = requests.get(url_cgi)
    assert resp.status_code == 200


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
    transactions = fimi.fetch_any('chess.dat', data_home='./skmine/datasets/tests/')
    assert os.path.isfile('./skmine/datasets/tests/chess.dat')
    os.remove('./skmine/datasets/tests/chess.dat')
    assert transactions.name == 'chess'
    assert len(transactions) == 3196
    assert all(type(item) == int for transaction in transactions for item in transaction)


def test_fetch_any_already_fetched():
    transactions = fimi.fetch_any('test_fimi_file_int.dat', data_home='./skmine/datasets/tests/')
    assert len(transactions) == 4
    assert all(type(item) == int for transaction in transactions for item in transaction)
    pd.testing.assert_series_equal(pd.Series([[1, 2, 3], [4, 5], [2], [8, 9]], name='test_fimi_file_int'), transactions)


def test_fetch_any_not_fetched_gz():
    transactions = fimi.fetch_any("iris.D19.N150.C3.num.gz", base_url="https://cgi.csc.liv.ac.uk/~frans/KDD/Software/LUCS-KDD-DN/DataSets/", data_home="./skmine/datasets/tests/")
    assert os.path.isfile('./skmine/datasets/tests/iris.D19.N150.C3.num.gz')
    os.remove('./skmine/datasets/tests/iris.D19.N150.C3.num.gz')
    assert transactions.name == 'iris.D19.N150.C3.num'
    assert len(transactions) == 150
    assert all(type(item) == int for transaction in transactions for item in transaction)
