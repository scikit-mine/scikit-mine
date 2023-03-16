import pytest

from skmine.periodic.run_mine import bronKerbosch3Plus, merge_cycle_lists


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


def test_recover_splits_rec():
    spoints = {
        (0, 2): None,
        (1, 3): None,
        (2, 4): None,
        (3, 5): -1,
        (0, 3): None,
        (1, 4): None,
        (2, 5): 4,
        (0, 4): None,
        (1, 5): 4,
        (0, 5): 4
    }
    ia = 0
    iz = 5
    depth = 0
    singletons=False

