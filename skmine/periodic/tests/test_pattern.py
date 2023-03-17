from unittest import mock

import numpy as np
import pytest

from skmine.periodic.pattern import Pattern, getEDict, codeLengthE, getEforOccs


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
                'parent': 0
            }
    }


@pytest.fixture
def tree_data_complex():
    return {
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


def test_getOccsStar_complex(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    timestamp_event_pairs = pattern.getOccsStar()
    expected_output = [
        (0, 4, '0,0'), (8643, 4, '0,1'), (17286, 4, '0,2'), (25929, 4, '0,3'), (34572, 4, '0,4'),
        (0, 7, '1,0'), (8643, 7, '1,1'), (17286, 7, '1,2'), (25929, 7, '1,3'), (34572, 7, '1,4'),
        (50, 8, '2,0;0,0'), (50050, 8, "2,0;0,1"), (100050, 8, "2,0;0,2"), (8693, 8, "2,1;0,0"), (58693, 8, "2,1;0,1"),
        (108693, 8, "2,1;0,2"), (17336, 8, "2,2;0,0"), (67336, 8, "2,2;0,1"), (117336, 8, "2,2;0,2"),
        (25979, 8, "2,3;0,0"), (75979, 8, '2,3;0,1'), (125979, 8, '2,3;0,2'), (34622, 8, '2,4;0,0'),
        (84622, 8, '2,4;0,1'), (134622, 8, '2,4;0,2')
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
    oStar = [(0, 0, '0,0'), (8643, 0, '0,1'), (17286, 0, '0,2'), (25929, 0, '0,3'), (34572, 0, '0,4')]
    E = [-9, -3, 3, 9]
    expected_output = {'0,0': 0, '0,1': -9, '0,2': -3, '0,3': 3, '0,4': 9}
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
    expected_output = [158702220, 158702220 + 8643 - 9, 158702220 + 17286 - 9 - 3, 158702220 + 25929 - 9 - 3 + 3,
                       158702220 + 34572 - 9 - 3 + 3 + 9]
    assert res == expected_output


def test_getCovSeq(tree_data):
    pattern = Pattern(tree_data)
    with mock.patch.object(pattern, 'getOccsStar',
                           return_value=[(0, 0, '0,0'), (8643, 0, '0,1'), (17286, 0, '0,2'), (25929, 0, '0,3'),
                                         (34572, 0, '0,4')]):
        t0 = 158702220
        E = [-9, -3, 3, 9]
        res = pattern.getCovSeq(t0, E)
        expected_output = [(0 + 158702220, 0), (8643 + 158702220 - 9, 0), (17286 + 158702220 - 9 - 3, 0),
                           (25929 + 158702220 - 9 - 3 + 3, 0), (34572 + 158702220, 0)]
        assert expected_output == res


def test_getNbLeaves(tree_data_complex):
    pattern = Pattern(tree_data_complex)

    assert pattern.getNbLeaves() == 3
    assert pattern.getNbLeaves(3) == 1


def test_getNbOccs():
    pattern = Pattern({0: {'p': 8640, 'r': 7, 'children': [(1, 0), (2, 978)], 'parent': None},
                       1: {'parent': 0, 'event': 27}, 2: {'parent': 0, 'event': 10}})
    assert pattern.getNbOccs() == 14


def test_getDepth(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getDepth() == 2
    assert pattern.getDepth(3) == 1


def test_getWidth(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getWidth() == 3
    assert pattern.getWidth(3) == 1


def test_getAlphabet(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getAlphabet() == {4, 7, 8}
    assert pattern.getAlphabet(3) == {8}


def test_isSimpleCycle(tree_data):
    pattern = Pattern(tree_data)
    assert pattern.isSimpleCycle()


def test_isSimpleCycle_complex(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert not pattern.isSimpleCycle()
    assert pattern.isSimpleCycle(3)


def test_isNested(tree_data):
    pattern = Pattern(tree_data)
    assert not pattern.isNested()


def test_isNested_complex(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert not pattern.isNested()  # False because the width must be equal to 1


def test_isNested_true():
    pattern = Pattern(event=1, r=5, p=20)
    pattern.repeat(10, 500)
    assert pattern.isNested()


def test_isConcat(tree_data):
    pattern = Pattern(tree_data)
    assert not pattern.isConcat()
    pattern.append(2, 0)
    assert pattern.isConcat()


def test_getTypeStr(tree_data):
    pattern = Pattern(tree_data)
    assert pattern.getTypeStr() == "simple"
    pattern.append(2, 0)
    assert pattern.getTypeStr() == "concat"


def test_getTypeStr_nested():
    pattern = Pattern(event=1, r=5, p=20)
    pattern.repeat(10, 500)
    assert pattern.getTypeStr() == "nested"


def test_getTypeStr_other(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getTypeStr() == "other"


def test_codeLengthEvents():
    pattern = Pattern({0: {'p': 8643, 'r': 5, 'children': [(1, 0)], 'parent': None}, 1: {'event': 0, 'parent': 0}})
    adjFreqs = {"(": 1 / 3, ")": 1 / 3, 0: 1 / 3}
    assert pattern.codeLengthEvents(adjFreqs, nid=0) == -np.log2(adjFreqs["("]) - np.log2(adjFreqs[")"]) - \
           np.log2(adjFreqs[0])


def test_getMinOccs():
    pattern = Pattern(event=4, r=100, p=500)
    pattern.append(5, 50)
    nbOccs = {4: 3, 5: 7, -1: 10.0}
    min_occs = []
    res = pattern.getMinOccs(nbOccs, min_occs)
    assert res == [3]


def test_getRVals():
    pattern = Pattern(event=4, r=100, p=500)
    pattern.append(5, 50)
    pattern.getRVals() == [100, 50]


def test_codeLengthR():
    pattern = Pattern(event=4, r=100, p=500)
    pattern.append(5, 50)
    pattern.append(6, 15)
    nbOccs = {4: 100, 5: 50, 6: 15, -1: 165}
    assert pattern.codeLengthR(nbOccs) == np.log2(15)


def test_card0(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.cardO() == 25
    assert pattern.cardO(3) == 3


def test_codeLengthE():
    E = [5, 6, 7, 8]
    L_E = codeLengthE(E)
    assert L_E == 2 * len(E) + 5 + 6 + 7 + 8


def test_codeLengthPTop():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 8}})
    deltaT = 10932
    L_P = pattern.codeLengthPTop(deltaT=deltaT)
    assert L_P == np.log2((deltaT - 0) / (3 - 1))  # by default sigma(E)=0


def test_codeLengthT0():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 8}})
    deltaT = 10932
    EC_za = 0
    L_tau = pattern.codeLengthT0(deltaT, EC_za=EC_za)
    assert L_tau == np.log2(deltaT - EC_za - (3 - 1) * 150 + 1)


def test_hasNestedPDs_if(tree_data, tree_data_complex):
    # if: checks if the node id (pattern) has multiple children
    p1 = Pattern(tree_data)
    assert not p1.hasNestedPDs()

    p2 = Pattern(tree_data_complex)
    assert p2.hasNestedPDs()  # 3 children so this pattern has inter-block (inter-child distances) distances
    assert not p2.hasNestedPDs(3)


def test_hasNestedPDs_else():
    # if the node id has only one child it checks if it has nested periods
    pattern = Pattern(event=1, r=5, p=20)
    assert not pattern.hasNestedPDs()
    pattern.repeat(10, 500)
    assert pattern.hasNestedPDs()


def test_getCycleRs(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getCycleRs() == [5, 3]
    assert pattern.getCycleRs(3) == [3]


def test_getCyclePs(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getCyclePs() == [8643, 50000]
    assert pattern.getCyclePs(3) == [50000]


def test_canFactorize(tree_data_complex):
    pattern = Pattern({
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0), (2, 50)],
                'parent': None
            },
        1:
            {
                'p': 45,
                'r': 2,
                'children': [(3, 0)],
                'parent': None
            },
        2:
            {
                'p': 45,
                'r': 2,
                'children': [(4, 0)],
                'parent': None
            },
        3:
            {
                'event': 7,
                'parent': 1
            },
        4:
            {
                'event': 8,
                'parent': 2
            }
    })
    res = pattern.canFactorize()
    assert res == [0]  # the child nodes of 0 have the same period and repetition and only one child
    # The tree can be simplified


def test_factorizeTree():
    pattern = Pattern({
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0), (2, 50)],
                'parent': None
            },
        1:
            {
                'p': 45,
                'r': 2,
                'children': [(3, 0)],
                'parent': None
            },
        2:
            {
                'p': 45,
                'r': 2,
                'children': [(4, 0)],
                'parent': None
            },
        3:
            {
                'event': 7,
                'parent': 1
            },
        4:
            {
                'event': 8,
                'parent': 2
            }
    })
    pattern.factorizeTree()  # id 2 is lost during the factorization
    assert pattern.nodes == {
        0:
            {
                'p': 8643,
                'r': 5,
                'children': [(1, 0)],
                'parent': None
            },
        1:
            {
                'p': 45,
                'r': 2,
                'children': [(3, 0), (4, 50)],
                'parent': None
            },
        3:
            {
                'event': 7,
                'parent': 1
            },
        4:
            {
                'event': 8,
                'parent': 1
            }
    }


def test_nodeP(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.nodeP(10) == 0  # this node doesn't exist
    assert pattern.nodeP(0) == 8643
    assert pattern.nodeP(1) == 0  # a leaf
    assert pattern.nodeP(2) == 0  # a leaf
    assert pattern.nodeP(3) == 50000
    assert pattern.nodeP(4) == 0


def test_nodeR(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.nodeR(10) == 0  # this node doesn't exist
    assert pattern.nodeR(0) == 5
    assert pattern.nodeR(1) == 0  # a leaf
    assert pattern.nodeR(2) == 0  # a leaf
    assert pattern.nodeR(3) == 3
    assert pattern.nodeR(4) == 0


def test_getMajorOccs():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 9}})
    occs = [159626862, 159626934, 159627012, 159627084, 159627162, 159627234]
    assert pattern.getMajorOccs(occs) == occs[::2]  # get only the timestamps of the first event 8


def test_pattKey_simple(tree_data):
    pattern = Pattern(tree_data)
    assert pattern.pattKey() == "[5,8643](4)"


def test_pattKey_complex(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.pattKey() == "[5,8643](4-0-7-50-[3,50000](8))"


def test_pattMinorKey():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 9}})
    assert pattern.pattMinorKey() == "8-[72]-9"
    pattern.repeat(7, 100000)
    assert pattern.pattMinorKey() == "[3,150](8-72-9)"


def test_pattMinorKey_tree_complex_data():
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
    assert pattern.pattMinorKey() == "4-[0]-7-[50]-[3,50000](8)"


def test_pattMajorKey():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 9}})
    assert pattern.pattMajorKey() == "[3,150]"
    assert pattern.pattMajorKey(nid=1) == "[]"
    assert pattern.pattMajorKey(nid=2) == "[]"
    pattern.repeat(7, 100000)
    assert pattern.pattMajorKey() == "[7,100000]"
    assert pattern.pattMajorKey(nid=3) == "[3,150]"


def test_pattMajorKey_list():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 9}})
    assert pattern.pattMajorKey_list() == [3, 150]
    assert pattern.pattMajorKey_list(nid=1) == []
    assert pattern.pattMajorKey_list(nid=2) == []
    pattern.repeat(7, 100000)
    assert pattern.pattMajorKey_list() == [7, 100000]
    assert pattern.pattMajorKey_list(nid=3) == [3, 150]


def test_pattern___str__(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.__str__() == "[r=5 p=8643](4 [d=0] 7 [d=50] [r=3 p=50000](8))"

    assert pattern.__str__(leaves_first=True) == "(4 [d=0] 7 [d=50] (8)[r=3 p=50000])[r=5 p=8643]"

    assert pattern.__str__(map_ev={4: "wake up", 7: "coffee", 8: "shower"}) == "[r=5 p=8643](wake up [d=0] coffee [" \
                                                                               "d=50] [r=3 p=50000](shower))"


def test_getTreeStr(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getTreeStr() == "|_ [0] r=5 p=8643\n\t| d=0\n\t|_ [1] 4\n\t| d=0\n\t|_ [2] 7\n\t| d=50\n\t|_ [3] " \
                                   "r=3 p=50000\n\t\t| d=0\n\t\t|_ [4] 8\n"


def test_getEventsMinor(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getEventsMinor() == [4, 7, 8, 8, 8]
    assert pattern.getEventsMinor(nid=3) == [8]
    assert pattern.getEventsMinor(rep=True) == [4, 7, 8, 8, 8] * 5


def test_getEventsList(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getEventsList() == ['(', '4', '7', '(', '8', ')', ')']
    assert pattern.getEventsList(nid=3) == ['(', '8', ')']
    assert pattern.getEventsList(add_delimiter=False) == ['4', '7', '8']


def test_getKeyLFromNid():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 8}, 2: {'parent': 0, 'event': 9}})
    assert pattern.getKeyLFromNid(nid=0) == []
    assert pattern.getKeyLFromNid(nid=1) == [(0, 0)]  # second element in the tuple is 0 because by default rep=0
    assert pattern.getKeyLFromNid(nid=2) == [(1, 0)]
    assert pattern.getKeyLFromNid(nid=2, rep=-1) == [(1, 2)]  # second element in the tuple is r of the parent - 1


def test_getKeyLFromNid_complex(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getKeyLFromNid(nid=3, rep=-1) == [(2, 4)]
    assert pattern.getKeyLFromNid(nid=2, rep=-1) == [(1, 4)]
    assert pattern.getKeyLFromNid(nid=4, rep=-1) == [(2, 4), (0, 2)]
    assert pattern.getKeyLFromNid(nid=1, rep=-1) == [(0, 4)]


def test_getKeyFromNid(tree_data_complex):
    pattern = Pattern(tree_data_complex)
    assert pattern.getKeyFromNid(nid=4, rep=-1) == "2,4;0,2"


def test_getEforOccs():
    map_occs = {1: 2, 2: 3, 4: 6}
    occs = [(1, 0.1, 1), (4, 0.2, 2), (5, 0.3, 3)]
    expected_output = [-1, 1, 0]
    assert getEforOccs(map_occs, occs) == expected_output


def test_codeLength():
    pattern = Pattern({0: {'p': 150, 'r': 3, 'children': [(1, 0), (2, 72)], 'parent': None},
                       1: {'parent': 0, 'event': 3}, 2: {'parent': 0, 'event': 3}})
    t0 = 159626862
    E = [0, 0, 0, 0, 0]
    data_details = {'t_start': 159626160, 't_end': 159627738, 'deltaT': 1578,
                    'nbOccs': {0: 1, 1: 1, 2: 3, 3: 22, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
                               14: 2, 15: 1, 16: 2, 17: 1, 18: 1, -1: 44.0},
                    'orgFreqs': {0: 0.022727272727272728, 1: 0.022727272727272728, 2: 0.06818181818181818, 3: 0.5,
                                 4: 0.022727272727272728, 5: 0.022727272727272728, 6: 0.022727272727272728,
                                 7: 0.022727272727272728, 8: 0.022727272727272728, 9: 0.022727272727272728,
                                 10: 0.022727272727272728, 11: 0.022727272727272728, 12: 0.022727272727272728,
                                 13: 0.022727272727272728, 14: 0.045454545454545456, 15: 0.022727272727272728,
                                 16: 0.045454545454545456, 17: 0.022727272727272728, 18: 0.022727272727272728},
                    'adjFreqs': {'(': 0.3333333333333333, ')': 0.3333333333333333, 0: 0.007575757575757576,
                                 1: 0.007575757575757576, 2: 0.022727272727272728, 3: 0.16666666666666666,
                                 4: 0.007575757575757576, 5: 0.007575757575757576, 6: 0.007575757575757576,
                                 7: 0.007575757575757576, 8: 0.007575757575757576, 9: 0.007575757575757576,
                                 10: 0.007575757575757576, 11: 0.007575757575757576, 12: 0.007575757575757576,
                                 13: 0.007575757575757576, 14: 0.015151515151515152, 15: 0.007575757575757576,
                                 16: 0.015151515151515152, 17: 0.007575757575757576, 18: 0.007575757575757576},
                    'blck_delim': 3.1699250014423126}
    code_length = pattern.codeLength(t0, E, data_details)
    np.testing.assert_almost_equal(code_length, 58.1, decimal=1)


