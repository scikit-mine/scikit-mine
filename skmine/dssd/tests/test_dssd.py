import numpy as np
import pandas
import pytest
from ..subgroup import Subgroup
from ..description import Description
from ..cond import Cond
from ..dssd import apply_dominance_pruning, fixed_size_compression_beam_selection, fixed_size_cover_selection, fixed_size_description_selection,  multiplicative_weighted_covering_score, update_topk, var_size_compression_beam_selection, var_size_cover_selection, var_size_description_selection, mine
from ..custom_types import ColumnType


def test_update_topk():
    cand1 = Subgroup(Description(), .5, pandas.Index([]))
    cand2 = Subgroup(Description(), 2.0, pandas.Index([1, 5]))
    cand3 = Subgroup(Description(), 3.1, pandas.Index([1, 5]))

    max_length = 3
    cands = []

    assert update_topk(cands, cand2, max_length) == [cand2]

    assert update_topk(cands, cand1, max_length) == [cand2, cand1]

    assert update_topk(cands, cand3, max_length) == [cand3, cand2, cand1]

    assert update_topk(cands, Subgroup(Description()), max_length) == [cand3, cand2, cand1]
    assert len(cands) == 3


def test_dominance_pruning():
    def mock_quality(cand: Subgroup):
        length = len(cand.description.conditions)
        return 0 if length == 0 else cand.quality

    def mock_cover(cand: Subgroup):
        return pandas.DataFrame(index=cand.cover)


    candidate = Subgroup(Description())
    apply_dominance_pruning(candidate, quality_func = mock_quality, cover_func = mock_cover)

    assert candidate.description == Subgroup(Description()).description

    candidate = Subgroup(Description([
        Cond("num", "<", 2.0),
        Cond("num", ">", 2.0),
        Cond("num", "<", 4.0),
        Cond("num", ">", 4.0),
    ]), 5.)

    apply_dominance_pruning(candidate, quality_func = mock_quality, cover_func = mock_cover)

    print(candidate.description)
    assert candidate.description == Subgroup(Description([Cond("num", ">", 4.0)])).description


def test_multiplicative_weighted_covering_score():
    empty_cand = Subgroup(Description([]), cover=pandas.Index([]))

    # ensure only non empty candidates can be given as argument
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score(empty_cand, [], 0.9)

    # ensure a non empty candidate is given as an argument
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score(empty_cand, [], 0)

    # ensure invalid weigh raises an exception
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score(Subgroup(Description([]), cover=pandas.Index([0])), [], 1.5)

    # non empty cover with empty selection
    assert multiplicative_weighted_covering_score(Subgroup(Description([]), 0.0, cover=pandas.Index([0, 2])), [], .9) == 1

    # watch score decrease as the selection already contains transactions covered by the candidate
    cand1 = Subgroup(Description([]), cover=pandas.Index([0]))
    assert multiplicative_weighted_covering_score(cand1, [cand1], 0.9) == .9
    assert multiplicative_weighted_covering_score(cand1, [cand1, cand1], 0.9) == .9 ** 2

    # candidate with a cover size different that those already in the selection
    # candidate size is taken into account while computing the score 
    cand2 = Subgroup(Description([]), cover=pandas.Index([0, 5, 6]))
    assert multiplicative_weighted_covering_score(cand2, [cand1, cand1], 0.9) == (.9 ** 2 + 1 + 1) / len(cand2.cover)


def test_unimplemented_methods():
    with pytest.raises(NotImplementedError):
        fixed_size_compression_beam_selection([], [])

    with pytest.raises(NotImplementedError):
        var_size_compression_beam_selection([])


res = None
def test_mining():
    df = pandas.DataFrame({
        'num': [1, 1, 1.9, 2, 2, 3, 3, 4, 4, 4],
        'cat': ["find_me", "find_me", "find_me_too_please", *(["regular", "regular-kindof"] * 3), "regular"],
        'ts': [
                    [1, 4, 1, 4, 1, 3, 1], # <-- very different time series
                    [1, 4, 1, 4, 1, 3, 1], # <-- very different time series
                    [1, 4, 1, 4, 1, 3, 1], # <-- very different time series
                    [1, 1, 1, 1, 1, 1, 1], # <-- very similar completely flat
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]
                ]
    })

    column_types = {"num": ColumnType.NUMERIC, "cat": ColumnType.NOMINAL, "ts": ColumnType.TIME_SERIE}
    # d = DSSDEMM(df, column_types)
    global res
    res = mine(df, column_types, descriptive_attributes=["cat", "num"], model_attributes=["ts"], max_depth=5, k = 10, j = 1000, beam_width=10, min_cov=1, num_cut_points={"num": 5}, #{}, 
    quality_measure="ts_quality", 
    quality_parameters={"model_attribute": "ts", "target_model": "eub", "dist_measure": "euclidean"},
    selection_strategy="cover",
    selection_params={"weight": .9},
    refinement_operator_name="basic", save_intermediate_results=True)

    print(res)
