import numpy as np
import pandas as pd
import pytest

from ..utils import (
    _check_growth_rate,
    _check_min_supp,
    _check_random_state,
    bron_kerbosch,
    filter_maximal,
    filter_minimal,
    intersect2d,
)


def test_check_random_state():
    random_state = np.random.RandomState(18)
    assert random_state == _check_random_state(random_state)

    assert isinstance(_check_random_state(4), np.random.RandomState)


def test_check_random_state_error():
    with pytest.raises(ValueError):
        _check_random_state(object())


def test_wrong_minimum_supports():
    wrong_values = [-1, -100, 2.33, 150.55]
    for wrong_supp in wrong_values:
        with pytest.raises(ValueError):
            _check_min_supp(wrong_supp)

    with pytest.raises(TypeError):
        _check_min_supp("string minimum support")

    with pytest.raises(ValueError):
        _check_min_supp(12, accept_absolute=False)


# def test_minimum_support():
#     assert _check_min_supp(0.1) == 0.1
#     assert _check_min_supp(10) == 10


def test_wrong_growth_rate():
    for wrong_gr in [0.3, -10]:
        with pytest.raises(ValueError):
            _check_growth_rate(wrong_gr)


def test_growth_rate():
    assert _check_growth_rate(1.5) == 1.5


def test_filter_max():
    D = pd.Series([{2, 3}, {2}, {4, 1}, {4, 7}, {4, 1, 8}])
    maximums = list(filter_maximal(D))

    assert maximums == D.iloc[[0, 3, 4]].tolist()


def test_filter_min():
    D = pd.Series([{2, 3}, {2}, {4, 1}, {4, 7}, {4, 1, 8}])
    maximums = list(filter_minimal(D))

    assert maximums == D.iloc[[1, 2, 3]].tolist()


def test_intersect2d():
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = [[1, 3, 5], [7, 1, 2], [4, 5, 6]]
    ab, a_ind, b_ind = intersect2d(a, b)
    np.testing.assert_array_equal(ab, np.array([a[1]]))
    np.testing.assert_array_equal(a_ind, np.array([1]))
    np.testing.assert_array_equal(b_ind, np.array([2]))


def test_bron_kerbosch():
    candidates = {
        "A": "BCE",
        "B": "ACDF",
        "C": "ABDF",
        "D": "CBEF",
        "E": "AD",
        "F": "BCD",
    }

    cliques = list(bron_kerbosch(candidates))

    assert cliques == [
        list("CBA"),
        list("FDCB"),
    ]
