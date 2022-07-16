import pandas
from ..selection_strategies import Cover
from ..subgroup import Subgroup
from ..description import Description
from ..cond import Cond
from ..dssd import DSSDMiner, apply_dominance_pruning, update_topk, mine
from ..quality_measures import EuclideanEub

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


def test_mining():
    df = pandas.DataFrame({
        'bin': [True, True, True, False,False,False,False,False,False,False],
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

    global res
    desc_attrs = ["bin", "cat", "num"]
    res = DSSDMiner(k=10, max_depth=5, j=1000, min_cov=1, quality=EuclideanEub(), selector=Cover()).fit(df[desc_attrs], df[["ts"]]).result
    print(res)
