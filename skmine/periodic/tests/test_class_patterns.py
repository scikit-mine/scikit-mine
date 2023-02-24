import pytest
import numpy as np

from skmine.periodic.class_patterns import _getChained, computePeriodDiffs, computePeriod, computeE, cost_one, \
    computeLengthEOccs, computeLengthResidual


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