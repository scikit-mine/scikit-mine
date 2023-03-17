import pytest

from skmine.periodic.candidate import Candidate
from skmine.periodic.pattern import Pattern


@pytest.fixture
def cand():
    cid = 1
    pattern = {
        'alpha': 1,
        'occs': [0, 51, 100],
        'p': 50,
        'cost': 50.22386061518789,
        'source': (2, 0)
    }
    return Candidate(cid, pattern)


@pytest.fixture
def cand_pattern():
    cid = 2
    pattern = Pattern(event=4, r=5, p=60480)
    O = [0, 60480, 120961, 181441, 241921]
    E = [0, 1, 0, 0]
    cost = 35
    return Candidate(cid, pattern, O, E, cost)


def test_Candidate_P_dict(cand):
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
    O = [0, 60480, 120961, 181441, 241921]
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


def test_getCost(cand):
    assert cand.getCost() == 50.22386061518789


def test_isPattern(cand, cand_pattern):
    assert cand.isPattern() is False
    assert cand_pattern.isPattern() is True


def test_getNbUOccs(cand, cand_pattern):
    assert cand.getNbUOccs() == 3  # cand.P is not a pattern so it returns the len of O
    # TODO : assert cand_pattern.getNbUOccs()


def test_getNbOccs(cand, cand_pattern):
    assert cand.getNbOccs() == 3
    assert cand_pattern.getNbOccs() == 5


def test_getEvOccs(cand, cand_pattern):
    assert cand.getEvOccs() == [(0, 1), (51, 1), (100, 1)]
    assert cand_pattern.getEvOccs() == [(0, 4), (60480, 4), (120961, 4), (181441, 4), (241921, 4)]


def test_getCostRatio(cand, cand_pattern):
    assert cand.getCostRatio() == cand.cost/3
    assert cand_pattern.getCostRatio() == cand_pattern.cost/5
    cand.cost = 0
    assert cand.getCostRatio() == 0


def test_getEvent(cand, cand_pattern):
    assert cand.getEvent() == 1
    assert cand_pattern.getEvent() == "4"


def test___str__(cand):
    assert cand.__str__() == "%f/%d=%f (%d)\t%s t0=%d" % (
            50.22386061518789, 3, cand.getCostRatio(), 3, {'alpha': 1, 'p': 50, 'source': (2, 0)}, 0)


# def test_adjustOccs(cand):
#     cand.adjustOccs()
#     print(cand.P)