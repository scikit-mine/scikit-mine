import pytest

from unittest import mock

from skmine.periodic.pattern import Pattern, getEDict


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


def test_getOccsStar_simple(tree_data):
    pattern = Pattern(tree_data)
    timestamp_event_pairs = pattern.getOccsStar()
    assert timestamp_event_pairs == [
        (0, 4, '0,0'), (8643, 4, '0,1'), (17286, 4, '0,2'), (25929, 4, '0,3'), (34572, 4, '0,4')
    ]


def test_getOccsStar_complex(tree_data):
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
        2:
            {
                'event': 7,
                'parent': 0
            },
        3:
            {
                'p': 50000,
                'r': 3,
                'children': [(4, 0)],
                'parent': 0
            },
        4:
            {
                'event': 8,
                'parent': 3
            }
    })
    timestamp_event_pairs = pattern.getOccsStar()
    expected_output = [
        (0, 4, '0,0'), (8643, 4, '0,1'), (17286, 4, '0,2'), (25929, 4, '0,3'), (34572, 4, '0,4'),
        (0, 7, '1,0'), (8643, 7, '1,1'), (17286, 7, '1,2'), (25929, 7, '1,3'), (34572, 7, '1,4'),
        (50, 8, '2,0;0,0'), (50050, 8, "2,0;0,1"), (100050, 8, "2,0;0,2"), (8693, 8, "2,1;0,0"), (58693, 8, "2,1;0,1"), (108693, 8, "2,1;0,2"), (17336, 8, "2,2;0,0"), (67336, 8, "2,2;0,1"), (117336, 8, "2,2;0,2"), (25979, 8, "2,3;0,0"), (75979, 8, '2,3;0,1'), (125979, 8, '2,3;0,2'), (34622, 8, '2,4;0,0'), (84622, 8, '2,4;0,1'), (134622, 8, '2,4;0,2')
    ]
    assert len(timestamp_event_pairs) == len(expected_output)
    assert set(timestamp_event_pairs) == set(expected_output)


def test_getTimesNidsRefs(tree_data):
    pattern = Pattern(tree_data)
    times_nids_refs = pattern.getTimesNidsRefs()
    expected_output = [
        (0, 1, '0,0'), (8643, 1, '0,1'), (17286, 1, '0,2'), (25929, 1, '0,3'), (34572, 1, '0,4')
    ]
    assert set(times_nids_refs) == set(expected_output)


def test_getEDict_with_non_empty_E():
    # Test case 1 : len(E) >= len(oStar) - 1
    oStar = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    E = [10, 11]
    expected_output = {3: 0, 6: 10, 9: 11}
    assert getEDict(oStar, E) == expected_output

    # Test case 2 : len(E) < len(oStar) - 1
    oStar = [(1,), (2,), (3,), (4,)]
    E = [10, 11]
    expected_output = {1: 0, 2: 0, 3: 0, 4: 0}
    assert getEDict(oStar, E) == expected_output


def test_getEDict_with_empty_E():
    # Test case 3 : len(E) < len(oStar) - 1
    oStar = [(1, 2), (3, 4)]
    E = []
    expected_output = {2: 0, 4: 0}
    assert getEDict(oStar, E) == expected_output


def test_getEDict_with_empty_oStar():
    # Test case 4
    oStar = []
    E = [10]
    expected_output = {}
    assert getEDict(oStar, E) == expected_output


def test_getCCorr(tree_data):
    pattern = Pattern()
    # we mock the result of gatherCorrKeys which is called in getCCorr
    with mock.patch.object(pattern, 'gatherCorrKeys', return_value=['0,0', '0,1']):
        result = pattern.getCCorr("0,2", {'0,0': 0, '0,1': -9, '0,2': -3, '0,3': 3, '0,4': 9})

        assert result == -12


def test_getOccs(tree_data):
    pattern = Pattern(tree_data)
    oStar = [(0, 0, '0,0'), (8643, 0, '0,1'), (17286, 0, '0,2'), (25929, 0, '0,3'), (34572, 0, '0,4')]
    t0 = 158702220
    E = {'0,0': 0, '0,1': -9, '0,2': -3, '0,3': 3, '0,4': 9}
    res = pattern.getOccs(oStar=oStar, t0=158702220, E=E)
    expected_output = [158702220, 158702220+8643-9, 158702220+17286-9-3, 158702220+25929-9-3+3, 158702220+34572-9-3+3+9]
    assert res == expected_output
