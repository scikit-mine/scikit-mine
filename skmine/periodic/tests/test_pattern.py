import pytest

from skmine.periodic.pattern import Pattern


@pytest.fixture
def tree_data():
    return {
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0)],
                'parent': None
            },
        1:
            {
                'event': 4,
                'parent': 0}
    }


def test_pattern(tree_data):
    pattern = Pattern(tree_data)

    assert pattern.next_id == 2
    assert pattern.nodes == tree_data


def test_pattern_event_r_p():
    pattern = Pattern(event=4, r=50, p=60480)

    assert pattern.next_id == 2
    assert pattern.nodes == {
        0: {
            'p': 60480,
            'r': 50,
            'children': [(1, 0)],
            'parent': None
        },
        1: {
            'event': 4,
            'parent': 0
        }
    }


def test_copy(tree_data):
    p1 = Pattern(tree_data)
    p2 = p1.copy()
    assert p1 != p2
    assert p2.next_id == p1.next_id
    assert p2.nodes == p1.nodes


def test_mapEvents(tree_data):
    pattern = Pattern(tree_data)
    pattern.mapEvents(["", "", "", "", "wake up", ""])
    assert pattern.nodes[1]["event"] == "wake up"


def test_getTranslatedNodes(tree_data):
    pattern = Pattern(tree_data)
    translated_nodes, offset_plus_len_map_nids, map_nids = pattern.getTranslatedNodes(offset=2)

    assert map_nids == {0: 2, 1: 3}
    assert offset_plus_len_map_nids == 4
    assert translated_nodes == {
        2:
            {
                'p': 8643,
                'r': 5,
                'children': [(3, 0)],
                'parent': None
            },
        3:
            {
                'event': 4,
                'parent': 2}
    }


def test_merge(tree_data):
    p1 = Pattern(tree_data)
    p2 = Pattern(event=1, r=100, p=500)
    map_nids = p1.merge(p2, d=5, anchor=0)
    assert map_nids == {1: 2, 0: 3}
    expected_nodes = {
        3:
            {
                'p': 500,
                'r': 100,
                'children': [(2, 0)],
                'parent': 0
            },
        2:
            {
                'event': 1,
                'parent': 3
            }

    }
    for node, value in expected_nodes.items():
        assert p1.nodes[node] == value


def test_append(tree_data):
    p1 = Pattern(tree_data)
    p1.append(event=2, d=300, anchor=0)
    assert p1.nodes == {
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0), (2, 300)],
                'parent': None
            },
        1:
            {
                'event': 4,
                'parent': 0
            },
        2:
            {
                'event': 2,
                'parent': 0
            }
    }
    assert p1.next_id == 3


def test_repeat(tree_data):
    pattern = Pattern(tree_data)
    pattern.repeat(50, 20000)

    # the root node exchanges its id 0 with the new root node of the tree
    assert pattern.nodes == {
        2:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0)],
                'parent': 0
            },
        1:
            {
                'event': 4,
                'parent': 2
            },
        0:
            {
                'p': 20000,
                'r': 50,
                'children': [(2, 0)],
                'parent': None
            }
    }
    assert pattern.next_id == 3


def test_isNode(tree_data):
    pattern = Pattern(tree_data)
    assert pattern.isNode(0)
    assert pattern.isNode(1)


def test_isInterm():
    pattern = Pattern({
        2:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0)],
                'parent': 0
            },
        1:
            {
                'event': 4,
                'parent': 2
            },
        0:
            {
                'p': 20000,
                'r': 50,
                'children': [(2, 0)],
                'parent': None
            }
    })
    assert pattern.isInterm(2)
    assert pattern.isInterm(1) is False
    assert pattern.isInterm(0)  # the root node is considered as intermediate


def test_isLeaf():
    pattern = Pattern({
        2:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0)],
                'parent': 0
            },
        1:
            {
                'event': 4,
                'parent': 2
            },
        0:
            {
                'p': 20000,
                'r': 50,
                'children': [(2, 0)],
                'parent': None
            }
    })
    assert pattern.isLeaf(2) is False
    assert pattern.isLeaf(1) is True
    assert pattern.isLeaf(0) is False


def test_getRightmostLeaves():
    pattern = Pattern({
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0), (2, 0), (3, 50)],
                'parent': None
            },
        1:
            {
                'event': 4,
                'parent': 0
            },
        3:
            {
                'event': 7,
                'parent': 0
            },
        2:
            {
                'p': 50000,
                'r': 10,
                'children': [(4, 0)],
                'parent': 0
            },
        4:
            {
                'event': 8,
                'parent': 3
            }
    })
    rightmost_nids = pattern.getNidsRightmostLeaves()
    rightmost_nids == [4, 3]  # based on the biggest id for each intermediate node

