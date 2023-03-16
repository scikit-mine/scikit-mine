import pytest

from skmine.periodic.candidate import Candidate
from skmine.periodic.pattern import Pattern


def test_Candidate_P_dict():
    cid = 1
    pattern = {
        'alpha': 1,
        'occs': [0, 51, 100],
        'p': 50,
        'cost': 50.22386061518789,
        'source': (2, 0)
    }
    cand = Candidate(cid, pattern)
    assert cand.cid == 1
    assert cand.cost == 50.22386061518789
    assert cand.O == [0, 51, 100]
    assert cand.P == {
        "alpha": 1,
        "p": 50,
        "source": (2, 0)
    }
    assert cand.E == [1, -1]
    assert cand.uncov is None
    assert cand.ev_occ is None


def test_Candidate_Pattern():
    cid = 2
    pattern = Pattern(event=4, r=5, p=60480)
    O = [i + i * 60480 for i in range(5)]
    E = [0, 1, 0, 0]
    cost = 35
    cand = Candidate(cid, pattern, O, E, cost)
    assert cand.P == pattern
    assert cand.O == O
    assert cand.E == E
    assert cand.cost == cost
    assert cand.uncov is None
    assert cand.ev_occ is None


def test_Candidate_P():
    cid = 1
    pattern = {
        'alpha': 1,
        'occs': [0, 51, 100],
        'p': 50,
        'cost': 50.22386061518789,
        'source': (2, 0),
        'P': {
            "alpha": 1,
            "p": 50,
            "source": (2, 0)
        }
    }
    cand = Candidate(cid, pattern)
    assert cand.cid == 1
    assert cand.cost == 50.22386061518789
    assert cand.O == [0, 51, 100]
    assert cand.P == {
        "alpha": 1,
        "p": 50,
        "source": (2, 0)
    }
    assert cand.E is None  # in this case, E is not automatically computed
    assert cand.uncov is None
    assert cand.ev_occ is None


def test_Candidate_p_is_None():
    cid = 1
    pattern = {
        'alpha': 1,
        'occs': [0, 51, 100],
        'cost': 50.22386061518789,
        'source': (2, 0),
    }
    cand = Candidate(cid, pattern)
    assert cand.cid == 1
    assert cand.cost == 50.22386061518789
    assert cand.O == [0, 51, 100]
    assert cand.P == {
        "alpha": 1,
        "p": 50,
        "source": (2, 0)
    }
    assert cand.E == [1, -1]
    assert cand.uncov is None
    assert cand.ev_occ is None
