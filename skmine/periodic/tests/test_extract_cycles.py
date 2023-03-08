import numpy as np
import pytest

from skmine.periodic.extract_cycles import combine_splits, compute_cycles_dyn


def test_combine_splits():
    splits1 = [(0, 2), (4, 5), (7, 10)]
    adj_splits1 = [(2, 3), (5, 6), (6, 8)]
    assert combine_splits(splits1, adj_splits1) == [(0, 3), (5, 6), (6, 8)]  # FIXME : to be explained

    splits2 = [(0, 1), (4, 5), (8, 9)]
    adj_splits2 = [(2, 3), (5, 6), (6, 8)]
    assert combine_splits(splits2, adj_splits2) == [(0, 1), (2, 3), (5, 6), (6, 8)]

    splits3 = [(0, 1), (4, 5), (8, 9)]
    adj_splits3 = []
    assert combine_splits(splits3, adj_splits3) == [(0, 1), (4, 5), (8, 9)]  # returns splits3 because adj_splits3 is
    # empty

    splits4 = [(0, 1), (4, 5), (8, 9)]
    adj_splits4 = [(1, 3), (5, 6), (7, 8)]
    assert combine_splits(splits4, adj_splits4) == [(0, 3), (5, 6), (7, 8)]
