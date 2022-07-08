import pandas
import pytest
from ..subgroup import Subgroup
from ..cond import Cond
from ..custom_types import ColumnType
from ..description import Description
from ..table import Table
from ..utils import _get_cut_points_smart
from .. import refinement_operators as ro


df = pandas.DataFrame({
    "num": [0, 4, 3.5, 6],
    "bin": [True, False] * 2,
    "a": ["one", "two"] * 2
})

dataset_size = len(df)

column_types = {"num": ColumnType.NUMERIC, "a": ColumnType.NOMINAL, "bin": ColumnType.BINARY}

min_coverage = 2

parent_candiate = Subgroup(Description(), cover=df.index)

def mock_cover(cand: Subgroup):
    length = len(cand.description.conditions)
    if length == 0: 
        return pandas.DataFrame(index=pandas.Index(range(dataset_size)))
    elif length == 1: 
        return pandas.DataFrame(index=pandas.Index(range(int(dataset_size/2))))
    else: # length == 2 | _ 
        return pandas.DataFrame(index=pandas.Index(range(int(dataset_size/4))))


def mock_quality(cand: Subgroup):
    return 0 if  len(cand.description.conditions) ==  0 else cand.quality


discretization = {"num": 2}
extra_parameters = {
    "dataset": Table(df, column_types),
    "num_cut_points": discretization,
    "cover_func": mock_cover,
    "quality_func": mock_quality,
    "min_cov": min_coverage
}


def test_dssd_paper_refinements():
    assert set(_get_cut_points_smart(sorted(df["num"]), num_cut_points=2)) == {3.75, 6}

    numeric_conditions = [Cond("num", "<", 3.75), Cond("num", ">", 3.75), Cond("num", "<", 6.0), Cond("num", ">", 6.0)]
    nominal_conditions = [Cond("a", "==", "one"), Cond("a", "!=", "one"), Cond("a", "==", "two"), Cond("a", "!=", "two")]
    binary_conditions = [Cond("bin", "==", True), Cond("bin", "==", False)]
    conds_store = [*numeric_conditions, *nominal_conditions, *binary_conditions]
    official_ro = ro.RefinementOperatorOfficial(**extra_parameters)


    # testing the append method
    new_cand1 = Subgroup(Description([Cond("", "<", "")]), cover=pandas.Index(range(3)), parent=parent_candiate)
    assert official_ro.check_and_append_candidate(new_cand1, []) == [new_cand1]
    assert official_ro.check_and_append_candidate(Subgroup(Description(), parent=parent_candiate), []) == []

    # update the 
    start_cand = Subgroup(Description())
    start_cand.cover = mock_cover(start_cand).index

    cands_step1 = official_ro.refine_candidate(start_cand, [])

    # ensure that there are as much generated candidates as conditions in the 
    # condition store. one candidate for every condition
    assert len(cands_step1) == len(conds_store)

    # ensure that all the newly generated candidates have 1 condition
    assert all(len(c.description) == 1 for c in cands_step1)

    # assert that for every single candidate in the resulting list
    # none has more than one equal condition on the same attribute
    for cand in cands_step1:
        for cond in cand.description.conditions:
            if cond.op == '==':
                assert all(c.attribute != cond.attribute or c == cond for c in cand.description.conditions)
 
    # ensure every subgroup cover has half cover length compared to the base dataset
    # as specified by the mock cover function
    assert all(len(c.cover) == dataset_size / 2 for c in cands_step1)

    # make sure each candidate is unique 
    assert len(cands_step1) == len(set(cands_step1))

    # ensure all expected binary/nominal/numeric candidates were generated
    assert {Subgroup(Description([cond])) for cond in binary_conditions}.issubset(cands_step1)
    assert {Subgroup(Description([cond])) for cond in nominal_conditions}.issubset(cands_step1)
    assert {Subgroup(Description([cond])) for cond in numeric_conditions}.issubset(cands_step1)

    # update the start cand and begin a new refinement operation
    start_cand = cands_step1[0]
    cands_step2 = official_ro.refine_candidate(Subgroup(Description([Cond("a", "!=", "one")])), [])

    # cands_step2 should have no element because of the min coverage that eleminated all of the potential candidates as 
    # the mock cover_func return a cover of size 1 for all candidate with two conditions in their description
    assert len(cands_step2) == 0

    assert official_ro._refine_binary(Subgroup(Description([Cond("bin", "==", True)])), "bin", []) == None
