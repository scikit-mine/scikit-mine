import dataclasses
from collections import Counter
from itertools import compress

import numpy as np
import pandas as pd
import pytest

from ..tree import (
    Node,
    PeriodicPatternMiner,
    Tree,
    combine_horizontally,
    combine_vertically,
    encode_leaves,
    get_occs,
    greedy_cover,
    grow_horizontally,
)


@pytest.mark.parametrize("tau", [0, 330])
def test_create_tree_3_wakeup_breakfast(tau):
    """
    during 3 weeks
    on every business day
    wake up at 7:00 AM
    breakfast at 7:10 AM

    periods and distances expressed in minutes
    """
    week_node = Node(
        r=5, p=1440, children=["wake up", "breakfast"], children_dists=[10]
    )
    tree = Tree(tau, r=3, p=1440 * 7, children=[week_node])

    assert dataclasses.is_dataclass(tree)

    instances = tree.to_list()

    assert len(instances) == tree._n_occs == 30  # 2 events per day, 5 days for 3 weeks

    assert instances[0][0] == tau  # first occurence at tau

    assert tree._size == len(tree) == 4

    assert {"r", "p"}.issubset(tree.to_dict().keys())


def test_prefit():
    logs = pd.Series(["wake up", "breakfast"] * 10)

    cm = PeriodicPatternMiner()
    singletons = cm.prefit(logs)

    assert all((t.r == 10 for t in singletons))
    assert all((t.p == 2 for t in singletons))


def test_node_equal():
    assert Node(r=3, p=2, children="b") == Node(r=3, p=2, children="b")
    assert Node(r=3, p=2, children="bc", children_dists=[2]) != Node(
        r=3, p=2, children="cd", children_dists=[2]
    )


def test_node_id():
    assert id(Node(r=3, p=2, children="b")) != id(Node(r=3, p=2, children="a"))
    n = Node(r=1, p=4)
    assert id(n) == id(n)


def test_combine_vertically():
    """ Inspired from fig.4.b) in the original paper """
    trees = [
        Tree(2, r=3, p=2, children="ce", children_dists=[1], E=[1, 0, 0, 1, 0]),
        Tree(13, r=3, p=2, children="ce", children_dists=[1], E=[0, 0, -2, 2, 0]),
        Tree(35, r=3, p=2, children="ce", children_dists=[1], E=[1, -1, 0, 1, -1]),
        Tree(26, r=3, p=2, children="ce", children_dists=[1], E=[0, 0, 0, 0, 0]),
        Tree(24, r=5, p=2),  # should not be combined, good `tau` but bad `r`
        Tree(96, r=3, p=2),  # should not be combined, good `r`, `p` but bad `tau`
        Tree(47, r=3, p=2, children="dc", children_dists=[1]),  # wrong children
    ]
    cv = combine_vertically(trees)
    assert len(cv) == 1
    T = cv[0]
    assert str(T) == "2 {r=4, p=11} ({r=3, p=2} (c - 1 - e))"
    first_node = T.children[0]
    assert str(first_node) == "{r=3, p=2} (c - 1 - e)"
    assert trees[0] in T.get_internal_nodes()  # assert ref is same

    assert T.E.tolist() == (
        trees[0].E.tolist()
        + [0]
        + trees[1].E.tolist()
        + [2]
        + trees[3].E.tolist()
        + [-2]
        + trees[2].E.tolist()
    )

    """
    assert [_[0] for _ in get_occs(T, tau=T.tau, E=T.E, sort=False)[::6]] == sorted(
        [t.tau for t in trees[:4]]
    )"""  # FIXME get_occs seems to be broken


def test_grow_horizontally():
    """see fig.4 b) from the original paper"""
    trees = [
        Tree(2, r=6, p=7, children=["wake up"]),
        Tree(4, r=5, p=7, children=["breakfast"]),
        Tree(5, r=5, p=7, children=["take metro"]),
    ]

    T = grow_horizontally(*trees)

    assert str(T) == "2 {r=5, p=7} (wake up - 2 - breakfast - 1 - take metro)"
    assert T._n_occs == 15  # merging 3 trees, minimum r is 5
    # TODO : check E


def test_combine_horizontally():
    V = [
        Tree(2, r=6, p=7, children="b"),
        Tree(4, r=5, p=7, children="a"),
        Tree(5, r=5, p=7, children=[Node(r=3, p=2, children="b")]),
        Tree(7, r=8, p=10, children="a"),
    ]

    H = combine_horizontally(V)
    # FIXME: all have been merged, because MDL cost compute is missing
    # last a should not be there
    assert str(H[0]) == "2 {r=5, p=7} (b - 2 - a - 1 - {r=3, p=2} (b) - 2 - a)"


def test_get_occs_singleton():
    t = Tree(tau=2, r=3, p=12, children="b", E=[-1, 1])
    assert t.get_occs() == [(2, "b"), (13, "b"), (26, "b")]


@pytest.mark.parametrize(
    "E, positions",
    [
        ([0] * 5, tuple(range(6))),
        ([0, 1, 0, 0, 2], (0, 1, 3, 4, 5, 8)),
        ([0, 1, -2, 0, 3], (0, 1, 3, 2, 5, 9)),
    ],
)
def test_get_occs(E, positions):
    node = Node(r=3, p=2, children="ab", children_dists=[1])
    pos, chars = zip(*get_occs(node, E, sort=False))
    assert "".join(chars) == "ababab"
    assert pos == positions


@pytest.mark.parametrize(
    "E, positions",
    [([1, 0, 0, 0, 1], (0, 4, 5, 5, 8, 10)), (None, (0, 3, 4, 5, 8, 9))],
)
def test_get_occs_complex(E, positions):
    node = Node(
        r=2, p=5, children=["b", Node(r=2, p=1, children="a")], children_dists=[3]
    )
    pos, chars = zip(*get_occs(node, E=E))
    assert "".join(chars) == "baabaa"
    assert pos == positions


def test_discover_simple():
    occs = [2, 5, 7, 13, 18, 21, 26, 30, 31]
    S = pd.Series(list("bacbacbac"), index=occs)
    ppm = PeriodicPatternMiner()
    ppm.fit(S)
    bigger, cost = ppm.codetable[0]

    assert str(bigger) == "2 {r=3, p=12} (b - 3 - a - 2 - c)"
    assert pytest.approx(cost, 14.51, abs=0.1)
    # assert bigger.get_occs() == list(zip(S.index, S))  # FIXME page 21

    rec_occs, rec_events = zip(*bigger.get_occs())
    assert list(rec_events) == S.values.tolist()
    # np.testing.assert_array_equal(S.index[bigger.tids], rec_occs)  # FIXME page 21
    occs_diff = np.abs(np.array(rec_occs) - np.array(occs))
    assert np.all(occs_diff <= 2)

    disc = ppm.discover()
    assert disc.dtypes.to_dict() == {
        "description": pd.np.dtype("object"),
        "cost": pd.np.dtype("float64"),
    }


def test_leaves_length():
    events = "bacbacbbac"
    occs = [2, 5, 7, 13, 18, 21, 26, 28, 30, 31]
    S = pd.Series(list(events), index=occs)
    ppm = PeriodicPatternMiner()
    ppm.fit(S)
    event_freqs = {k: v / len(events) for k, v in Counter(events).items()}
    length = encode_leaves(ppm.codetable[0][0], event_freqs)
    assert length == pytest.approx(12.72, rel=1e-2)

    # TODO try with
    occs = [2, 5, 7, 13, 18, 21, 24, 26, 30, 31]
    # we should extract 2 cycles for b, one with p=11 (covering 24)
    # and one with p=12 (covering 26)
    # the current version only extracts the first one, hence not allowing for
    # horizontal combination to be perfomed, because the p is different


def test_mdl_cost_A():
    T = Node(3, 2, children=[Node(10, 2, children="a"), "b"], children_dists=[0])
    assert round(T.mdl_cost_A(a=0.5, b=0.3), 2) == 9.08


def test_mdl_costs():
    T = Node(
        3,
        2,
        children=[Node(5, 2, children=["a", "b"], children_dists=[1]), "c"],
        children_dists=[2],
    )
    event_frequencies = dict(a=0.2, b=0.3, c=0.5)
    dS = 10  # difference between max occurence and min occurence
    cost_R = T.mdl_cost_R(**event_frequencies)
    assert round(cost_R, 2) == 3.32

    cost_A = T.mdl_cost_A(**event_frequencies)
    assert round(cost_A, 2) == 11.4

    cost_tau = T.mdl_cost_tau(dS)
    assert round(cost_tau, 2) == 1.58

    cost_p0 = T.mdl_cost_p0(dS)
    assert round(cost_p0, 2) == 1.0

    # TODO : cost of D


def test_greedy_cover(monkeypatch):
    # set a fixed mdl cost for easier testing
    monkeypatch.setattr(Tree, "mdl_cost", lambda self, D, dS, **_: 4)
    T1 = Tree(0, r=3, p=5, tids={0, 4, 8}, children="a")  # _n_occs is 3
    T2 = Tree(0, r=5, p=1, tids={0, 5, 10, 15, 20}, children="b")
    T3 = Tree(5, r=4, p=1, tids={5, 9, 13, 17}, children="c")
    T4 = Tree(2, r=2, p=1, tids={0, 5}, children="d")  # 0 and 5 will be covered by T2

    # T2 should be the first inserted, followed by T3, and finally T1
    cover = greedy_cover([T1, T2, T3, T4], D=pd.Series(), dS=None, k=3)
    assert cover == [T2, T3, T1]  # no T4 because k=3


def test_str():
    """check string formatting works correctly with a nested structure"""
    node = Node(
        r=5,
        p=7,
        children=[Node(r=3, p=2, children=["b", "c"], children_dists=[2]), "a"],
        children_dists=[4],
    )
    node_str = str(node)
    assert node_str == "{r=5, p=7} ({r=3, p=2} (b - 2 - c) - 4 - a)"

    tree = Tree(
        tau=3, **{k: v for k, v in node.__dict__.items() if not k.startswith("_")}
    )
    tree_str = str(tree)
    assert tree_str.endswith(node_str)

    assert eval(repr(node)) == node
    assert eval(repr(tree)) == tree

    assert Node.from_str(node_str) == node
    # assert Tree.from_str(tree_str) == tree  # TODO


def test_interactive():
    events = "bacbacbbac"
    occs = [2, 5, 7, 13, 18, 21, 26, 28, 30, 31]
    S = pd.Series(list(events), index=occs)
    answers = [True, False, True, True]

    ppm = PeriodicPatternMiner(k=2)
    candidates = ppm.generate_candidates(ppm.prefit(S))
    with pytest.warns(UserWarning):
        for cand in compress(candidates, answers):
            ppm.update(cand)

    assert len(ppm.discover()) == sum(answers)
