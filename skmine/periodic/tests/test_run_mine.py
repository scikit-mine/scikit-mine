import pytest

from skmine.periodic.run_mine import bronKerbosch3Plus


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
