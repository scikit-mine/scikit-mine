import dataclasses
from collections import Counter

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

    assert len(instances) == 30  # 2 events per day, 5 days for 3 weeks

    assert instances[0][0] == tau  # first occurence at tau

    assert tree.size() == 4

    assert tree.to_dict() == dict(
        r=3, p=1440 * 7, children=[week_node.to_dict()], children_dists=[],
    )


def test_prefit():
    logs = pd.Series(["wake up", "breakfast"] * 10)

    cm = PeriodicPatternMiner()
    singletons = cm._prefit(logs)

    assert all((t.r == 10 for t in singletons))
    assert all((t.p == 2 for t in singletons))


def test_node_equal():
    assert Node(r=3, p=2, children="b") == Node(r=3, p=2, children="b")
    assert Node(r=3, p=2, children="bc", children_dists=[2]) != Node(
        r=3, p=2, children="cd", children_dists=[2]
    )


def test_combine_vertically():
    """ Inspired from fig.4.b) in the original paper """
    trees = [
        Tree(2, r=3, p=2, children="ce", children_dists=[1]),
        Tree(13, r=3, p=2, children="ce", children_dists=[1]),
        Tree(35, r=3, p=2, children="ce", children_dists=[1]),
        Tree(26, r=3, p=2, children="ce", children_dists=[1]),
        Tree(24, r=5, p=2),  # should not be combined, good `tau` but bad `r`
        Tree(96, r=3, p=2),  # should not be combined, good `r`, `p` but bad `tau`
        Tree(47, r=3, p=2, children="dc", children_dists=[1]),  # wrong children
    ]
    cv = combine_vertically(trees)
    assert len(cv) == 1
    T = cv[0]
    assert T.tau == 2
    assert T.r == 4
    assert T.p == 11
    assert len(T.children) == 1
    first_node = T.children[0]
    assert isinstance(first_node, Node)
    assert not isinstance(first_node, Tree)
    assert first_node.r == trees[0].r
    assert first_node.p == trees[1].p
    assert trees[0] in T.get_internal_nodes()  # assert ref is same


def test_grow_horizontally():
    """see fig.4 b) from the original paper"""
    trees = [
        Tree(2, r=6, p=7, children=["wake up"]),
        Tree(4, r=5, p=7, children=["breakfast"]),
        Tree(5, r=5, p=7, children=["take metro"]),
    ]

    T = grow_horizontally(*trees)

    assert T.tau == 2
    assert T.r == 5
    assert T.p == 7
    assert T.children_dists == [2, 1]
    assert T.children == ["wake up", "breakfast", "take metro"]


def test_combine_horizontally():
    V = [
        Tree(2, r=6, p=7, children="b"),
        Tree(4, r=5, p=7, children="a"),
        Tree(5, r=5, p=7, children=[Node(r=3, p=2, children="b")]),
        Tree(7, r=8, p=10, children="a"),  # should not be included, wrong `p`
    ]

    H = combine_horizontally(V)
    assert H[0].tau == 2
    assert H[0].r == 5
    assert H[0].children_dists == [
        2,
        1,
        2,
    ]  # FIXME: all have been merged, because MDL cost compute is missing
    assert H[0].children[:2] == ["b", "a"]
    assert isinstance(H[0].children[2], Node)


def test_discover_simple():
    occs = [2, 5, 7, 13, 18, 21, 26, 30, 31]
    S = pd.Series(list("bacbacbac"), index=occs)
    ppm = PeriodicPatternMiner()
    ppm.fit(S)
    bigger = ppm.forest[0]
    assert bigger.tau == 2
    assert bigger.p == 12
    assert bigger.r == 3
    assert len(bigger.children) == 3
    assert bigger.children == ["b", "a", "c"]

    rec_occs, rec_events = zip(*bigger.get_occs())
    assert list(rec_events) == S.values.tolist()
    occs_diff = np.abs(np.array(rec_occs) - np.array(occs))
    assert np.all(occs_diff <= 2)


def test_leaves_length():
    events = "bacbacbbac"
    occs = [2, 5, 7, 13, 18, 21, 26, 28, 30, 31]
    S = pd.Series(list(events), index=occs)
    ppm = PeriodicPatternMiner()
    ppm.fit(S)
    event_freqs = {k: v / len(events) for k, v in Counter(events).items()}
    length = encode_leaves(ppm.forest[0], event_freqs)
    assert length == pytest.approx(12.72, rel=1e-2)

    # TODO try with
    occs = [2, 5, 7, 13, 18, 21, 24, 26, 30, 31]
    # we should extract 2 cycles for b, one with p=11 (covering 24)
    # and one with p=12 (covering 26)
    # the current version only extracts the first one, hence not allowing for
    # horizontal combination to be perfomed, because the p is different
