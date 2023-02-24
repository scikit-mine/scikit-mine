import pytest
import numpy as np

from skmine.periodic.class_patterns import _getChained, computePeriodDiffs, computePeriod, computeE


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
