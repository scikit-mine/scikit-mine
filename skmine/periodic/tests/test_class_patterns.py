import pytest
import numpy as np

from skmine.periodic.class_patterns import _getChained, computePeriodDiffs, computePeriod, computeE, cost_one, \
    computeLengthEOccs, computeLengthResidual, key_to_l, l_to_key, l_to_br, key_to_br, propCmp


def test__getChained_with_keys():
    listsd = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    }
    keys = ["a", "b"]

    result = _getChained(listsd, keys)
    assert list(result) == [1, 2, 3, 4, 5, 6]


def test__getChained_without_keys():
    listsd = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    }

    result = _getChained(listsd)
    assert list(result) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_computePeriodDiffs_odd():
    diffs = [-1.2, -0.7, 2.3, 5, 17.6]
    result = computePeriodDiffs(diffs)

    assert result == 2


def test_computePeriodDiffs_even():
    diffs = [25.7, -0.7, 5, 17.6]
    result = computePeriodDiffs(diffs)

    assert result == int((5 + 17.6) / 2)


def test_computePeriod_without_sort():
    occs = [-10, 5, 10, 12, 20]
    result = computePeriod(occs)
    assert result == 6


def test_computePeriod_with_sort():
    occs = [12, 5, 10, -10, 20]
    result = computePeriod(occs)
    assert result == -1

    result = computePeriod(occs, sort=True)
    assert result == 6


def test_computeE_without_sort():
    occs = [-6, 0, 5, 12]
    p0 = 5

    result = computeE(occs, p0)
    assert result == [1, 0, 2]


def test_computeE_with_sort():
    occs = [5, 0, -6, 12]
    p0 = 5

    result = computeE(occs, p0)
    assert result != [1, 0, 2]

    result = computeE(occs, p0, sort=True)
    assert result == [1, 0, 2]


def test_cost_one():
    minimalist_data_details = {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 2, -1: 8},
        "orgFreqs": {1: 0.75, 0: 0.25}
    }

    result = cost_one(minimalist_data_details, 1)
    assert -np.log2(0.75) + np.log2(60480 + 1) == result


def test_cost_one_missing_alpha():
    minimalist_data_details = {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 2, -1: 8},
        "orgFreqs": {1: 0.75, 0: 0.25}
    }

    result = cost_one(minimalist_data_details, 5)
    assert 0 + np.log2(60480 + 1) == result


def test_computeLengthEOccs():
    occs = [-6, 0, 5, 12]
    cp = 5

    result = computeLengthEOccs(occs, cp)
    assert 9 == result


def test_computeLengthEOccs_empty():
    occs = []
    cp = 5

    result = computeLengthEOccs(occs, cp)
    assert 0 == result


def test_computeLengthEOccs_single():
    occs = [1]
    cp = 5

    result = computeLengthEOccs(occs, cp)
    assert 0 == result


def test_computeLengthResidual():
    data_details = {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 2, -1: 8},
        "orgFreqs": {1: 0.75, 0: 0.25}
    }
    residual = {"alpha": 0, "occs": [0]}
    assert pytest.approx(17.88, 0.01) == computeLengthResidual(data_details, residual)


def test_computeLengthResidual_multiple_occs():
    data_details = {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 2, -1: 8},
        "orgFreqs": {1: 0.75, 0: 0.25}
    }
    residual = {"alpha": 0, "occs": [0, 1]}
    assert pytest.approx(35.76, 0.01) == computeLengthResidual(data_details, residual)


def test_key_to_l():
    # Test case 1: empty string
    assert key_to_l("") == []

    # Test case 2: valid input string
    assert key_to_l("1,2,3;4,5") == [[1, 2, 3], [4, 5]]

    # Test case 3: invalid input string
    assert key_to_l("1,2,3;a,b,c") == []

    # Test case 4: input string with leading/trailing spaces
    assert key_to_l(" 1,2,3 ; 4,5 ") == [[1, 2, 3], [4, 5]]

    # Test case 5: input string with leading/trailing separators
    assert key_to_l(";1,2,3;4,5;") == []


def test_l_to_key():
    l = [[1, 2], [4, 5]]
    assert "1,2;4,5" == l_to_key(l)

    # Test case 1: empty list
    assert l_to_key([]) == ""

    # Test case 2: valid input string
    assert l_to_key([[1, 2], [4, 5]]) == "1,2;4,5"

    # Test case 3: valid tuple
    assert l_to_key([(1, 2), (4, 5)]) == "1,2;4,5"

    # Test case 4: invalid with more than 2 elements in tuple
    with pytest.raises(TypeError):
        l_to_key([(1, 2, 3), (4, 5)]) == "1,2,3;4,5"


def test_l_to_br():
    # Test case 1: empty list
    assert l_to_br([]) == "B<>"

    # Test case 2: list with single element
    assert l_to_br([[0, 1]]) == "B1<2>"

    # Test case 3: list with multiple elements
    assert l_to_br([[0, 1], [1, 2], [3, 4]]) == "B1,2,4<2,3,5>"

    # Test case 4: list with leading/trailing spaces
    assert l_to_br([[0, 1], [1, 2], [3, 4], [5, 6]]) == "B1,2,4,6<2,3,5,7>"


def test_key_to_br():
    key = "1,2;4,5"
    assert key_to_br(key) == "B2,5<3,6>"


def test_propCmp_with_list():
    # Test with a list of properties
    props = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ]
    pid = 1
    expected_output = (3.0, 4.0, 5.0)
    assert propCmp(props, pid) == expected_output


def test_propCmp_with_ndarray():
    # Test with a numpy ndarray of properties
    import numpy as np
    props = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ])
    pid = 2
    expected_output = (6.0, 7.0, 8.0)
    assert propCmp(props, pid) == expected_output
