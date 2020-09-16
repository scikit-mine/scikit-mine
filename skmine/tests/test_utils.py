import pytest
import numpy as np
import pandas as pd
from ..utils import _check_random_state
from ..utils import _check_min_supp
from ..utils import _check_growth_rate
from ..utils import filter_maximal
from ..utils import filter_minimal


def test_check_random_state():
    random_state = np.random.RandomState(18)
    assert random_state == _check_random_state(random_state)

    assert isinstance(_check_random_state(4), np.random.RandomState)


def test_check_random_state_error():
    with pytest.raises(TypeError):
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


def test_minimum_support():
    assert _check_min_supp(0.1) == 0.1
    assert _check_min_supp(10) == 10


def test_wrong_growth_rate():
    for wrong_gr in [0.3, -10]:
        with pytest.raises(ValueError):
            _check_growth_rate(wrong_gr)


def test_growth_rate():
    assert _check_growth_rate(1.5) == 1.5


def test_filter_max():
    D = pd.Series(
        [
            {2, 3},
            {2},
            {4, 1},
            {4, 7},
            {4, 1, 8},
        ]
    )
    maximums = list(filter_maximal(D))

    assert maximums == D.iloc[[0, 3, 4]].tolist()


def test_filter_min():
    D = pd.Series(
        [
            {2, 3},
            {2},
            {4, 1},
            {4, 7},
            {4, 1, 8},
        ]
    )
    maximums = list(filter_minimal(D))

    assert maximums == D.iloc[[1, 2, 3]].tolist()
