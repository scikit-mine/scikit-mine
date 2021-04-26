import pandas as pd
import pytest

from ..tree import Node, PeriodicPatternMiner, Tree, combine_vertically


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

    instances = tree.tolist()

    assert len(instances) == 30  # 2 events per day, 5 days for 3 weeks

    assert instances[0][0] == tau  # first occurence at tau

    assert tree.size() == 4


def test_prefit():
    logs = pd.Series(["wake up", "breakfast"] * 10)

    cm = PeriodicPatternMiner()
    singletons = cm._prefit(logs)

    assert all((t.r == 10 for t in singletons))
    assert all((t.p == 2 for t in singletons))


def test_combine_vertically():
    """ Inspired from fig.4.b) in the original paper """
    trees = [
        Tree(2, r=3, p=2),  # TODO : add children
        Tree(13, r=3, p=2),
        Tree(35, r=3, p=2),
        Tree(26, r=3, p=2),
        Tree(24, r=5, p=2),  # should not be combined, good `tau` but bad `r`
        Tree(96, r=3, p=2),  # should not be combined, good `r`, `p` but bad `tau`
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
    assert first_node.r == trees[0].r
    assert first_node.p == trees[1].p
    assert trees[0] in T.get_internal_nodes()  # assert ref is same
