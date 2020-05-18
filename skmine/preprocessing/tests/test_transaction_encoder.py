from ..transaction_encoder import TransactionEncoder
import pandas as pd
import pytest

@pytest.mark.parametrize('sparse_output', [True, False])
def test_transform(sparse_output):
    D = [[1, 9, 2], [2, 3]]
    res = TransactionEncoder(sparse_output=sparse_output).fit_transform(D)

    assert res.columns.tolist() == [1, 2, 3, 9]
    pd.testing.assert_series_equal(
        res.sum(axis=0),
        pd.Series([1, 2, 1, 1], index=[1, 2, 3, 9])
    )

