import pandas as pd
import pytest

from ..bide import BIDE, invertedindex


@pytest.fixture
def data():  # https://github.com/chuanconggao/PrefixSpan-py/blob/master/README.md
    return [
        [0, 1, 2, 3, 4],
        [1, 1, 1, 3, 4],
        [2, 1, 2, 2, 0],
        [1, 1, 1, 2, 2],
    ]


# @pytest.mark.parametrize("min_supp", (2, ))
def test_fit_all(data):
    pass


def test_fit_top_k(data):
    pass


@pytest.mark.parametrize("min_supp", (2,))
def test_discover_all(data, min_supp):
    bide = BIDE(min_supp=min_supp, min_len=0).fit(data)
    patterns = bide.discover().to_dict()
    assert patterns == {
        (0,): 2,
        (1,): 4,
        (1, 2,): 3,
        (1, 2, 2): 2,
        (1, 3, 4): 2,
        (1, 1, 1): 2,
    }


def test_discover_top_k(data):
    pass


def test_inverted_index(data):
    ii = invertedindex(data)
    assert ii == {
        0: [(0, 0), (2, 4)],
        1: [(0, 1), (1, 0), (2, 1), (3, 0)],
        2: [(0, 2), (2, 0), (3, 3)],
        3: [(0, 3), (1, 3)],
        4: [(0, 4), (1, 4)],
    }


def test_fit_non_integers(data):
    data = [[str(_) for _ in t] for t in data]
    bide = BIDE(min_supp=2, min_len=0).fit(data)
    assert bide._db == data  # FIXME dataset should not be kept as ref
