import pytest
import numpy as np
from ..utils import lazydict
from ..utils import _check_random_state
from ..utils import _check_min_supp
from ..utils import _check_growth_rate

def test_lazydict():
    d = lazydict(lambda e: e*2)
    assert d[2] == 4
    assert 2 in d.keys()

def test_lazydict_no_default():
    d = lazydict()
    with pytest.raises(KeyError):
        d[3]

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
        _check_min_supp('string minimum support')

def test_minimum_support():
    assert _check_min_supp(.1) == .1
    assert _check_min_supp(10) == 10


def test_wrong_growth_rate():
    for wrong_gr in [.3, -10]:
        with pytest.raises(ValueError):
            _check_growth_rate(wrong_gr)


def test_growth_rate():
    assert _check_growth_rate(1.5) == 1.5