from ..transaction_encoder import TransactionEncoder
from ..transaction_encoder import make_vertical
import pandas as pd
import pytest


def test_make_vertical():
    D = ["ABC"] * 5 + ["AB", "A", "B"]
    standard_codetable = make_vertical(D)
    pd.testing.assert_series_equal(
        standard_codetable.map(len), pd.Series([7, 7, 5], index=["A", "B", "C"])
    )


def test_fit_wrong():
    with pytest.raises(TypeError):
        TransactionEncoder().fit(2)


@pytest.mark.parametrize("sparse_output", [True, False])
def test_transform(sparse_output):
    D = [[1, 9, 2], [2, 3]]
    res = TransactionEncoder(sparse_output=sparse_output).fit_transform(D)

    assert res.columns.tolist() == [1, 2, 3, 9]
    pd.testing.assert_series_equal(
        res.sum(axis=0), pd.Series([1, 2, 1, 1], index=[1, 2, 3, 9])
    )


@pytest.mark.parametrize("sparse_output", [True, False])
def test_transform_gen(sparse_output):
    D = [[1, 9, 2], [2, 3]]

    def gen():
        for t in D:
            yield t

    res = TransactionEncoder(sparse_output=sparse_output).fit_transform(gen())

    assert res.columns.tolist() == [1, 2, 3, 9]
    pd.testing.assert_series_equal(
        res.sum(axis=0), pd.Series([1, 2, 1, 1], index=[1, 2, 3, 9])
    )
