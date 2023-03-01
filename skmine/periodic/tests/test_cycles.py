import pytest

import pandas as pd

from skmine.periodic.cycles import _remove_zeros, _iterdict_str_to_int_keys


def test_remove_zeros():
    numbers = pd.Int64Index([1587022200000000000, 1587108540000000000, 1587194940000000000,
                             1587281400000000000, 1587367920000000000, 1587627000000000000], dtype='int64')
    expected_output = (pd.Int64Index([158702220, 158710854, 158719494, 158728140, 158736792, 158762700], dtype='int64'),
                       10)
    numbers_without_zeros, n_zeros = _remove_zeros(numbers)
    assert (expected_output[0] == numbers_without_zeros).all()
    assert expected_output[1] == n_zeros


def test_iterdict_str_to_int_keys_with_str_keys():
    assert _iterdict_str_to_int_keys({"1": {"2": "value1"}, "3": ["4", "5"]}) == {1: {2: "value1"}, 3: ["4", "5"]}


def test_iterdict_str_to_int_keys_with_mixed_keys():
    assert _iterdict_str_to_int_keys({"1": {"key_2": "value1"}, 3: ["4", "5"]}) == {1: {"key_2": "value1"}, 3: ["4", "5"]}


def test_iterdict_str_to_int_keys_with_nested_dicts():
    assert _iterdict_str_to_int_keys({"1": {"2": {"3": "value1"}}}) == {1: {2: {3: "value1"}}}


def test_iterdict_str_to_int_keys_with_nested_lists():
    assert _iterdict_str_to_int_keys({"1": {"2": ["3", {"4": "value1"}]}}) == {1: {2: ["3", {4: "value1"}]}}
