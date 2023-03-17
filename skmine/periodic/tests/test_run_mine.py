import numpy
import pytest

from skmine.periodic.candidate import Candidate
from skmine.periodic.candidate_pool import CandidatePool
from skmine.periodic.run_mine import bronKerbosch3Plus, merge_cycle_lists, makeCandOnOrder, get_top_p, \
    prepare_tree_nested


def test_bronKerbosch3Plus():
    #  /1-5
    # 0 |x|
    # |\2-3
    # \   /
    #   4
    graph = {
        0: {1, 2},
        1: {0, 2, 3, 5},
        2: {0, 1, 3, 5},
        3: {1, 2, 4, 5},
        4: {0, 3},
        5: {1, 2, 3}
    }
    collect = []
    P = {0, 1, 2, 3, 4, 5}
    R = None
    X = None
    bronKerbosch3Plus(graph=graph, collect=collect, P=P, R=R, X=X)
    assert collect == [{0, 1, 2}, {1, 2, 3, 5}]


def test_merge_cycle_lists():
    cyclesL = [
        [{'alpha': 7,
          'occs': [159627012, 159687498, 159747978],
          'p': 60483,
          'cost': 52.149166914953845,
          'source': (0, 0)}],
        [],
        [{'alpha': 7,
          'occs': [159627012, 159687498, 159747978],
          'p': 60483,
          'cost': 52.14909723805441}]
    ]
    cycles = merge_cycle_lists(cyclesL)
    assert cycles == [
        {'alpha': 7,
         'occs': [159627012, 159687498, 159747978],
         'p': 60483,
         'cost': 52.14909723805441,
         'source': (2, 0)}
    ]


def test_get_top_p():
    occ_ordc = [(2, 5), (1, 10), (1, 7), (3, 3), (2, 8)]
    assert get_top_p(occ_ordc) == (0, 1, 2)

    occ_ordc = [(2, 5), (1, 10), (1, 7), (1, 3), (2, 8)]
    assert get_top_p(occ_ordc) == (0, 1, 3)

    occ_ordc = [(2, 5), (1, 10), (1, 7), (1, 11), (2, 8)]
    assert get_top_p(occ_ordc) == (0, 3, 1)


def test_prepare_tree_nested():
    prds = [5640, 150]
    lens = [3, 4]
    cand = Candidate(
        cid=23,
        P={'alpha': 9, 'p': 150, 'pos': [66, 68, 69, 70], 'uncov': {66, 68, 69, 70}, 'source': (1, 25)},
        O=[159652698, 159652842, 159652992, 159653142],
        E=[-6, 0, 0],
        cost=54.65572817232237
    )
    tree = prepare_tree_nested(cand, prds, lens)
    assert tree == {
        0: {'p': 5640, 'r': 3, 'children': [(1, 0)], 'parent': None},
        1: {'p': 150, 'r': 4, 'children': [(2, 0)], 'parent': 0},
        2: {'event': 9, 'parent': 1}
    }
