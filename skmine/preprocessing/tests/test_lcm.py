import pandas as pd
import pytest
import numpy as np

from ..lcm import LCM

D = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 5],
    [2, 5],
    [1, 2, 4, 5, 6],
    [2, 4],
    [1, 4, 6],
    [3, 4, 6],
]

true_item_to_tids = {
    1 : {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3 : {0, 1, 6},
    4 : {0, 3, 4, 5, 6},
    5 : {0, 1, 2, 3},
    6 : {0, 3, 5, 6},
}

true_patterns = pd.DataFrame([  # from D with min_supp=3
        [{2}, 5],
        [{4}, 5],
        [{2, 4}, 3],
        [{2, 5}, 4],
        [{4, 6}, 4],
        [{1, 4, 6}, 3],
        [{3}, 3],
], columns=['itemset', 'support'])

true_patterns.loc[:, 'itemset'] = true_patterns.itemset.map(tuple)


def test_lcm_fit():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]

def test_first_parent_limit_1():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    limit = 1
    tids = lcm.item_to_tids[limit]

    ## pattern = {4, 6} -> first parent OK
    itemset, tids = next(lcm._inner(frozenset([4, 6]), tids, limit), (None, None))
    assert itemset == (1, 4, 6)
    assert len(tids) == 3

    # pattern = {} -> first parent fails
    itemset, tids = next(lcm._inner(frozenset(), tids, limit), (None, None))
    assert itemset == None


def test_first_parent_limit_2():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    # pattern = {} -> first parent OK
    tids = lcm.item_to_tids[2]
    itemset, tids = next(lcm._inner(frozenset(), tids, 2), (None, None))
    assert itemset == (2,)
    assert len(tids) == 5

    # pattern = {4} -> first parent OK
    tids = lcm.item_to_tids[2] & lcm.item_to_tids[4]
    itemset, tids = next(lcm._inner(frozenset([4]), tids, 2), (None, None))
    assert itemset == (2, 4)
    assert len(tids) == 3


def test_first_parent_limit_3():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    tids = lcm.item_to_tids[3]
    itemset, tids = next(lcm._inner(frozenset(), tids, 3), (None, None))
    assert itemset == (3,)
    assert len(tids) == 3

def test_first_parent_limit_4():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    tids = lcm.item_to_tids[4]
    itemset, tids = next(lcm._inner(frozenset(), tids, 4), (None, None))
    assert itemset == (4,)
    assert len(tids) == 5

def test_first_parent_limit_5():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    tids = lcm.item_to_tids[5]
    itemset, tids = next(lcm._inner(frozenset(), tids, 5), (None, None))
    assert itemset == (2, 5)
    assert len(tids) == 4


def test_first_parent_limit_6():
    lcm = LCM(min_supp=3)
    lcm._fit(D)

    tids = lcm.item_to_tids[6]
    itemset, tids = next(lcm._inner(frozenset(), tids, 6), (None, None))
    assert itemset == (4, 6)
    assert len(tids) == 4

def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_discover(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

    # 2. test with empty data
    lcm = LCM(min_supp=3)
    res = lcm.fit_discover([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_lcm_discover():
    lcm = LCM(min_supp=3)
    patterns = lcm.fit_discover(D)  # get new pattern set

    for itemset, true_itemset in zip(patterns.itemset, true_patterns.itemset):
        assert itemset == true_itemset
    pd.testing.assert_series_equal(patterns.support, true_patterns.support, check_dtype=False)


def test_lcm_transform():
    lcm = LCM(min_supp=3)
    X = lcm.fit_transform(D)
    assert X.columns.tolist() == list(range(1, 7))
    assert X.shape == (7, 6)  # 7 transactions, 6 items
    assert np.array_equal(
        np.unique(X.values),
        np.array([0, 3, 4])  # all supports of 5 have been overriden
    )


def test_relative_support_errors():
    wrong_values = [-1, -100, 2.33, 150.55]
    for wrong_supp in wrong_values:
        with pytest.raises(ValueError):
            LCM(min_supp=wrong_supp)

    with pytest.raises(TypeError):
        LCM(min_supp='string minimum support')


def test_relative_support():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm._fit(D)
    np.testing.assert_almost_equal(lcm._min_supp, 2.8, 2)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]
