import pandas
import pytest
from ..subgroup import Subgroup
from ..description import Description
from ..utils import column_shares, diff_items_count, _get_cut_points, _get_cut_points_smart, remove_duplicates, sort_subgroups, subgroup
from ..cond import Cond

def test_get_cut_points():
    res = _get_cut_points(lo=0, hi=10, num_cut_points=4)
    # ensure the number of cut points is 1 less than the number of bins count
    assert len(res) == 4

    assert res == [2, 4, 6, 8]

    with pytest.raises(ValueError):
        # make sure a bin count of zero raises an exception
        _get_cut_points(0, 5, num_cut_points=0)


def test_get_cut_points_smart():
    # ensure the number of cut points is 1 less than the number of bins count
    assert list(_get_cut_points_smart(values=[1, 5, 6, 6, 8], num_cut_points=2)) == [(5 + 6) / 2, (6 + 8)/2]

    assert list(_get_cut_points_smart(values=[5, 6], num_cut_points=2)) == [(5 + 6) / 2]


def test_diff_items_count():
    # test edge case: both collections being empty
    assert diff_items_count([], []) == 0

    l1 = [1, 5, 6, 2]
    # testing edge case: only one empty collection
    assert diff_items_count([], l1) == len(l1)
    assert diff_items_count(l1, []) == len(l1)

    # testing regular same length non empty collections
    assert diff_items_count([1, 2, 3, 6], [1, 3, 5, 2]) == 1

    l2 = [5, 7, 4, 2, 3]
    # testing different length non empty collections
    assert diff_items_count(l1, l2) == diff_items_count(l2, l1) == 3

def test_sort_subgroups():
    cand1 = Subgroup(Description(), .5, pandas.Index([])) 
    cand2 = Subgroup(Description(),  5, pandas.Index([]))
    cand3 = Subgroup(Description(), 65, pandas.Index([]))
    cands = [cand2, cand3, cand1]
    cands_descending = [cand3, cand2, cand1]
    cands_ascending  = [cand1, cand2, cand3]

    sort_subgroups(cands, descending=True)
    assert cands == cands_descending

    cands = [cand2, cand3, cand1]
    sort_subgroups(cands, descending=False)
    assert cands == cands_ascending 


def test_remove_duplicates():
    assert set(remove_duplicates([])) == set()
    assert set(remove_duplicates([1, 1])) == {1}
    assert set(remove_duplicates([1, 2, 3, 3, 2])) == {1, 2, 3}

    cand1 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))
    cand2 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))
    cands_list = [cand1, cand2]
    assert cand1 == cand2
    cands_set = remove_duplicates(cands_list)
    assert cand1 in cands_set
    assert cand2 in cands_set
    assert len(cands_set) == 1


def test_column_shares():
    df = pandas.DataFrame({
        "a": ["train", "train", "test", "train"],
        "b": [True, False, False, True]
    })

    shares = {
        "a": {"train": 3/4, "test": 1/4},
        "b": {True: 2/4, False: 2/4}
    }

    # testing edge case: empty columns 
    assert column_shares(df, []) == {}

    # ensuring the dataframe columns are used if no columns is provided
    assert column_shares(df, ["a", "b"]) == column_shares(df)

    # ensuring the returned result corresponds to the actual share of the values
    assert column_shares(df, ["a", "b"]) == shares

    res = column_shares(df, ["a", "b"])
    # ensuring the function actually return a 2 level default dictionnary
    assert res["c"] == {} # first level default dictionnary
    assert res["c"]["non-existent"] == 0 # second level dictionnary


def test_subgroup():
    df = pandas.DataFrame({
        "a": list(range(10)),
        "b": list(range(0, 20, 2)),
        "c": list(range(0, 30, 3)),
        "ts": [list(range(-5, 20, 5)) for _ in range(10)]
    })
    
    # ensure an empty subgroup description actually selects the entire dataset
    assert subgroup(df, Description()).equals(df.copy())

    res_df = pandas.DataFrame({
        "a": list(range(6, 10)),
        "b": list(range(12, 20, 2)),
        "c": list(range(18, 30, 3)),
        "ts": [list(range(-5, 20, 5)) for _ in range(4)]
    }, index=list(range(6, 10)))
    found_subgroup = subgroup(df, Description([Cond("a", ">", 5), Cond("a", "<", 10)]))
    print(res_df)
    print(found_subgroup)
    # ensure subgroup selection is accurate according to conditions
    assert res_df.equals(found_subgroup)

    df = pandas.DataFrame({
        "a": [1, 2, 3, 4, 5], 
        "bin": [True, False, False, False, True],
        "cat": ["t", "t", "t", "T", "T"]
    })

    res_df = pandas.DataFrame({"a": [5], "bin": [True], "cat": ["T"]}, index=[4])
    found_subgroup = subgroup(df, Description([Cond("a", ">", 3), Cond("a", "<", 10), Cond("bin", "==", True), Cond("cat", "==", "T")] ))
    print(res_df)
    print(found_subgroup)
    assert res_df.equals(found_subgroup)

