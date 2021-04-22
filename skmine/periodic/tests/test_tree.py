import pytest

from ..tree import Node, Tree


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
