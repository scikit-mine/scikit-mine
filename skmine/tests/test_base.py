import pandas as pd
import pytest

from ..base import BaseMiner, MDLOptimizer


def test_inst_params():
    class MyMiner(BaseMiner):
        def __init__(self, eps=3):
            self.eps = eps
            self._a = 2

        def fit(self, D):
            self._a = 12

        discover = lambda self: pd.DataFrame()

    kwargs = dict(eps=4)
    miner = MyMiner(**kwargs)
    assert miner.get_params() == kwargs  # get_params returns only unprotected attributes so only eps

    kwargs.update(eps=10)
    miner.set_params(**kwargs)
    assert miner.get_params() == kwargs

    assert miner.set_params().get_params() == kwargs  # stay untouched

    with pytest.raises(ValueError):
        miner.set_params(random_key=2)


def test_inst_params_no_init():
    class MyMiner(BaseMiner):
        def fit(self, D, y=None):
            return self

        discover = lambda self: pd.DataFrame()

    miner = MyMiner()
    assert miner.get_params() == dict()  # compared to the previous test, eps is not an attribute of the class


def test_mdl_repr():
    class A(MDLOptimizer):
        def __init__(self):
            self.codetable_ = {1: [0, 1], 2: [1]}

        def fit(self):
            return self

        def evaluate(self):
            return True

        def generate_candidates(self):
            return list()

        discover = lambda self: pd.Series(self.__dict__)

    a = A()

    assert isinstance(a._repr_html_(), str)

    assert isinstance(a.fit()._repr_html_(), str)
