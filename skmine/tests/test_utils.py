import pytest
import numpy as np
from ..utils import lazydict
from ..utils import _check_random_state

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
