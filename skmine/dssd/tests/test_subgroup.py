import pandas
from ..subgroup import Subgroup
from ..cond import Cond
from ..description import Description

def test_subgroup_creation():
    cand1 = Subgroup(Description(), 0.0, pandas.Index([]))

    # ensuring default arguments are set properly
    assert cand1.description == Description()
    assert cand1.quality == 0.0
    assert cand1.parent is None
    assert cand1.cover.identical(pandas.Index([]))

    cand1 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))

    # test the properties of the newly birthed candidate to make sure they fit the specification/description
    cand_child = cand1.child_with_new_condition(Cond("a", "<", 2.0))
    assert cand_child != cand1
    assert cand_child.quality == 0
    assert cand_child.parent == cand1
    assert cand_child.description == Description([Cond("a", "<", 2.0)])
    assert cand_child.cover.identical(pandas.Index([]))

def test_subgroup_comparison():
    cand1 = Subgroup(Description(), 1.0, pandas.Index([]))
    cand2 = Subgroup(Description([Cond("a", "==", "one")]), 2.0, pandas.Index([1, 5]))
    cand3 = Subgroup(Description([Cond("a", "==", "one")]), .5, pandas.Index([1, 5]))

    # comparisons based on quality
    assert cand1 <  cand2
    assert cand1 <= cand2
    assert cand2 >  cand1
    assert cand2 >= cand1

    assert cand1 <= cand1 and cand2 <= cand2
    assert cand1 >= cand1 and cand2 >= cand2

    assert not cand2 < cand1

    # equality based on description
    assert cand1 != cand2

    assert cand2 == cand3


def test_subgroup_hash():
    cand1 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))
    cand2 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))

    assert hash(cand1) == hash(cand2)

    cands_list = [cand1, cand2]

    assert cand1 == cand2
    cands_set = set(cands_list)

    assert cand1 in cands_set
    assert cand2 in cands_set
    assert len(cands_set) == 1


def test_subgroup_to_string():
    assert str(Subgroup(Description())) is not None
