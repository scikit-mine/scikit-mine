import pandas as pd
import pytest

from skmine.itemsets import LCM

D = pd.Series([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 5],
    [2, 5],
    [1, 2, 4, 5, 6],
    [2, 4],
    [1, 4, 6],
    [3, 4, 6],
])

true_item_to_tids = {
    1 : {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3 : {0, 1, 6},
    4 : {0, 3, 4, 5, 6},
    5 : {0, 1, 2, 3},
    6 : {0, 3, 5, 6},
}


def test_lcm_fit():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]

def test_first_parent_limit_1():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    limit = 1
    tids = lcm.item_to_tids[limit]

    ## pattern = {4, 6} -> first parent OK
    itemset, supp = next(lcm._inner(frozenset([4, 6]), tids, limit), (None, None))
    assert itemset == frozenset([1, 4, 6])
    assert supp == 3

    # pattern = {} -> first parent fails
    itemset, supp = next(lcm._inner(frozenset(), tids, limit), (None, None))
    assert itemset == None
    assert supp == None


def test_first_parent_limit_2():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    # pattern = {} -> first parent OK
    tids = lcm.item_to_tids[2]
    itemset, supp = next(lcm._inner(frozenset(), tids, 2), (None, None))
    assert itemset == frozenset([2])
    assert supp == 5

    # pattern = {4} -> first parent OK
    tids = lcm.item_to_tids[2] & lcm.item_to_tids[4]
    itemset, supp = next(lcm._inner(frozenset([4]), tids, 2), (None, None))
    assert itemset == frozenset([2, 4])
    assert supp == 3


def test_first_parent_limit_3():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    tids = lcm.item_to_tids[3]
    itemset, supp = next(lcm._inner(frozenset(), tids, 3), (None, None))
    assert itemset == frozenset([3])
    assert supp == 3

def test_first_parent_limit_4():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    tids = lcm.item_to_tids[4]
    itemset, supp = next(lcm._inner(frozenset(), tids, 4), (None, None))
    assert itemset == frozenset([4])
    assert supp == 5

def test_first_parent_limit_5():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    tids = lcm.item_to_tids[5]
    itemset, supp = next(lcm._inner(frozenset(), tids, 5), (None, None))
    assert itemset == frozenset([2, 5])
    assert supp == 4


def test_first_parent_limit_6():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    tids = lcm.item_to_tids[6]
    itemset, supp = next(lcm._inner(frozenset(), tids, 6), (None, None))
    assert itemset == frozenset([4, 6])
    assert supp == 4

def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_transform(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

    # 2. test with empty data
    lcm = LCM(min_supp=3)
    res = lcm.fit_transform([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_lcm_transform():
    lcm = LCM(min_supp=3)
    X = lcm.fit_transform(D)  # get new pattern set

    true_X = pd.DataFrame([
            [{2}, 5],
            [{4}, 5],
            [{2, 4}, 3],
            [{2, 5}, 4],
            [{4, 6}, 4],
            [{1, 4, 6}, 3],
            [{3}, 3],
    ], columns=['itemset', 'support'])

    true_X.loc[:, 'itemset'] = true_X.itemset.map(frozenset)

    for itemset, true_itemset in zip(X.itemset, true_X.itemset):
        assert itemset == true_itemset
    pd.testing.assert_series_equal(X.support, true_X.support, check_dtype=False)


def test_relative_support_errors():
    wrong_values = [-1, -100, 2.33, 150.55]
    for wrong_supp in wrong_values:
        with pytest.raises(ValueError):
            LCM(min_supp=wrong_supp)

    with pytest.raises(TypeError):
        LCM(min_supp='string minimum support')


def test_relative_support():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D)

    for item in lcm.item_to_tids.keys():
        assert set(lcm.item_to_tids[item]) == true_item_to_tids[item]
