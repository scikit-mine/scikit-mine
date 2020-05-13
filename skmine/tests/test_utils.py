import pytest
from ..utils import lazydict

def test_lazydict():
    d = lazydict(lambda e: e*2)
    assert d[2] == 4
    assert 2 in d.keys()

def test_lazydict_no_default():
    d = lazydict()
    with pytest.raises(KeyError):
        d[3]
