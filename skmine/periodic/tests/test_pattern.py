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

