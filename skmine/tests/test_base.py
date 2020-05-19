import pytest

from ..base import BaseMiner

def test_inst_params():
    class MyMiner(BaseMiner):
        def __init__(self, eps=3):
            self.eps = eps
            self._a = 2

        def fit(self, D):
            self._a = 12

    kwargs = dict(eps=4)
    miner = MyMiner(**kwargs)
    assert miner.get_params() == kwargs

    kwargs.update(eps=10)
    miner.set_params(**kwargs)
    assert miner.get_params() == kwargs

    assert miner.set_params().get_params() == kwargs  # stay untouched

    with pytest.raises(ValueError):
        miner.set_params(random_key=2)

def test_inst_params_no_init():
    class MyMiner(BaseMiner):
        def fit(self, D, y=None): return self

    miner = MyMiner()
    assert miner.get_params() == dict()
