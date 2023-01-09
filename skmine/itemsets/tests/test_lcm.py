import os

import pandas as pd
import pytest
import numpy as np

from ..lcm import LCM
from ..lcm import LCMMax

D = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 5],
    [2, 5],
    [1, 2, 4, 5, 6],
    [2, 4],
    [1, 4, 6],
    [3, 4, 6],
]

initial_item_to_tids = {
    1: {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3: {0, 1, 6},
    4: {0, 3, 4, 5, 6},
    5: {0, 1, 2, 3},
    6: {0, 3, 5, 6},
}

truer_reorderfreq_item_to_tids = {
    0: {0, 1, 2, 3, 4},
    1: {0, 3, 4, 5, 6},
    2: {0, 1, 2, 3},
    3: {0, 3, 5, 6},
    4: {0, 3, 5},
    5: {0, 1, 6},

}

true_patterns = pd.DataFrame(
    [  # from D with min_supp=3
        [{2}, 5],
        [{4}, 5],
        [{2, 4}, 3],
        [{2, 5}, 4],
        [{4, 6}, 4],
        [{1, 4, 6}, 3],
        [{3}, 3],
    ],
    columns=["itemset", "support"],
)

true_patterns.loc[:, "itemset"] = true_patterns.itemset.map(set)

NULL_RESULT = (None, None, 0)


def test_lcm_fit():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    for item in lcm.item_to_tids_.keys():
        assert set(lcm.item_to_tids_[item]) == truer_reorderfreq_item_to_tids[item]


# def test_first_parent_limit_1():
#     lcm = LCM(min_supp=3)
#     # lcm.return_tids = True
#     lcm.fit(D)
#
#     limit = 1
#     tids = lcm.item_to_tids_[limit]
#
#     # pattern = [3, 5] -> first parent OK
#     itemset, _, tids = next(lcm._inner((frozenset([3, 5]), tids), limit), NULL_RESULT)
#     assert set(itemset) == {3, 4, 6}
#     assert len(tids) == 5

    # pattern = [] -> first parent fails
    # itemset, _, _ = next(lcm._inner((frozenset(), tids), limit), NULL_RESULT)
    #
    # assert itemset == [4]


# def test_first_parent_limit_2():
#     lcm = LCM(min_supp=3)
#     lcm.return_tids = True
#     lcm.fit(D)
#
#     # pattern = [] -> first parent OK
#     tids = lcm.item_to_tids_[2]
#     itemset, _, tids = next(lcm._inner((frozenset(), tids), 2), NULL_RESULT)
#
#     assert set(itemset) == {2, 5}
#     assert len(tids) == 4
#
#     # pattern = [4] -> first parent OK
#     tids = lcm.item_to_tids_[2] & lcm.item_to_tids_[4]
#     itemset, _, tids = next(lcm._inner((frozenset([4]), tids), 2), NULL_RESULT)
#
#     assert itemset is None


# def test_first_parent_limit_3():
#     lcm = LCM(min_supp=3)
#     lcm.return_tids = True
#     lcm.fit(D)
#
#     tids = lcm.item_to_tids_[3]
#     itemset, _, tids = next(lcm._inner((frozenset(), tids), 3), NULL_RESULT)
#
#     assert set(itemset) == {4, 6}
#     assert len(tids) == 4


# def test_first_parent_limit_4():
#     lcm = LCM(min_supp=3)
#     lcm.return_tids = True
#     lcm.fit(D)
#
#     tids = lcm.item_to_tids_[4]
#     itemset, _, tids = next(lcm._inner((frozenset(), tids), 4), NULL_RESULT)
#
#     assert set(itemset) == {1, 4, 6}
#     assert len(tids) == 3
#
#
# def test_first_parent_limit_5():
#     lcm = LCM(min_supp=3)
#     lcm.return_tids = True
#     lcm.fit(D)
#
#     tids = lcm.item_to_tids_[5]
#     itemset, _, tids = next(lcm._inner((frozenset(), tids), 5), NULL_RESULT)
#
#     assert itemset == [3]
#     assert len(tids) == 3
#
#
# def test_first_parent_limit_6():
#     lcm = LCM(min_supp=3)
#     lcm.return_tids = True
#     lcm.fit(D)
#
#     tids = lcm.item_to_tids_[1]
#     itemset, _, tids = next(lcm._inner((frozenset(), tids), 1), NULL_RESULT)
#
#     assert itemset == [4]
#     assert len(tids) == 5


def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_discover(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

    # # 2. test with empty data  # !!!  not with check_estimators ensure_min_samples
    # lcm = LCM(min_supp=3)
    # res = lcm.fit_discover([])
    # assert isinstance(res, pd.DataFrame)
    # assert res.empty


def test_lcm_discover():
    lcm = LCM(min_supp=3)
    patterns = lcm.fit_discover(D)  # get new pattern set

    for itemset, true_itemset in zip(patterns.itemset, true_patterns.itemset):
        assert set(itemset) == true_itemset
    pd.testing.assert_series_equal(
        patterns.support, true_patterns.support, check_dtype=False
    )


def test_relative_support():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D)
    np.testing.assert_almost_equal(lcm.min_supp_, 2.8, 2)

    for item in lcm.item_to_tids_.keys():
        assert set(lcm.item_to_tids_[item]) == truer_reorderfreq_item_to_tids[item]


def test_database_containing_item_0():
    db = [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1],
        [0]
    ]
    db_true_patterns = pd.DataFrame(
        [  # from db with min_supp=4
            [{0}, 4]
        ],
        columns=["itemset", "support"],
    )
    db_true_patterns.loc[:, "itemset"] = db_true_patterns.itemset.map(set)

    lcm = LCM(min_supp=4)
    patterns = lcm.fit_discover(db)

    for itemset, true_itemset in zip(patterns.itemset, db_true_patterns.itemset):
        assert set(itemset) == true_itemset
    pd.testing.assert_series_equal(patterns.support, db_true_patterns.support, check_dtype=False)


def test_lcm_max():
    lcm = LCMMax(min_supp=3)
    patterns = lcm.fit_discover(D, return_tids=True)
    pattern_to_tuple = {tuple(k) for k in patterns.itemset}
    assert pattern_to_tuple == {
        (1, 4, 6),
        (2, 5),
        (2, 4),
        (3,),
    }


def test_lcm_lexicographic_order():
    db = [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1],
        [0]
    ]
    lcm = LCM(min_supp=3)
    patterns = lcm.fit_discover(D, lexicographic_order=False)
    longest_itemset = max(patterns.itemset, key=len)
    assert longest_itemset != [1, 4, 6]

    patterns = lcm.fit_discover(D, lexicographic_order=True)
    longest_itemset = max(patterns.itemset, key=len)
    assert longest_itemset == [1, 4, 6]


# def test_lcm_file_output():
#     lcm = LCM(min_supp=3)
#     lcm.fit_discover(D, lexicographic_order=True, out="./skmine/itemsets/tests/lcm_out.dat")
#     with open('./skmine/itemsets/tests/lcm_out.dat', 'r') as f:
#         lines = f.read().splitlines()
#     nb_itemsets = 0
#     true_itemsets = true_patterns.itemset.tolist()
#
#     for line, true_itemset in zip(lines, true_itemsets):
#         nb_itemsets += 1
#         itemset = line.split(") ")[1].split(" ")
#         assert len(itemset) == len(true_itemset)
#         for item in itemset:
#             assert int(item) in true_itemset
#     assert len(true_itemsets) == nb_itemsets
#     os.remove('./skmine/itemsets/tests/lcm_out.dat')


def test_max_length_lcm():
    lcm = LCM(min_supp=3)

    patterns = lcm.fit_discover(D, lexicographic_order=True)
    longest_itemset = max(patterns['itemset'], key=len)
    assert longest_itemset == [1, 4, 6]
    assert len(patterns) == len(true_patterns)

    patterns = lcm.fit_discover(D, lexicographic_order=True, max_length=2)
    longest_itemset = max(patterns['itemset'], key=len)
    assert longest_itemset != [1, 4, 6]
    assert len(patterns) == len(true_patterns)-1


def test_max_length_lcm_max():
    lcm = LCMMax(min_supp=3)

    patterns = lcm.fit_discover(D, lexicographic_order=True)
    longest_itemset = max(patterns['itemset'], key=len)
    # print("patterns 1", patterns)
    assert longest_itemset == [1, 4, 6]
    assert len(patterns) == 4

    patterns = lcm.fit_discover(D, lexicographic_order=True, max_length=2)
    longest_itemset = max(patterns['itemset'], key=len)
    # print("patterns 2", patterns)
    assert longest_itemset != [1, 4, 6]
    assert len(patterns) == 3

