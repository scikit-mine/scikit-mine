import numpy as np

from skmine.periodic.pattern_collection import _replace_list_in_list


def test__replace_list_in_list():
    # Test case 1
    l1 = [[1, 2, np.int64(3)], [4, 5]]
    i1 = 0
    expected_output1 = [[1, 2, 3], [4, 5]]
    assert _replace_list_in_list(l1, i1) == expected_output1

    # Test case 2
    l2 = [[1, 2, 3], [4, 5, np.int64(6)]]
    i2 = 0
    expected_output2 = [[1, 2, 3], [4, 5, np.int64(6)]]
    assert _replace_list_in_list(l2, i2) == expected_output2

    # Test case 3: index out of range
    l3 = [[1, 2, 3], [4, 5, 6]]
    i3 = 2
    expected_output3 = [[1, 2, 3], [4, 5, 6]]
    assert _replace_list_in_list(l3, i3) == expected_output3

    # Test case 4: empty list
    l4 = []
    i4 = 0
    expected_output4 = []
    assert _replace_list_in_list(l4, i4) == expected_output4


