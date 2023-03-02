import json
from unittest.mock import patch, mock_open

import numpy as np
import pytest

import pandas as pd
import datetime as dt

from skmine.datasets import fetch_health_app
from skmine.periodic.cycles import _remove_zeros, _iterdict_str_to_int_keys, PeriodicPatternMiner


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
    assert _iterdict_str_to_int_keys({"1": {"key_2": "value1"}, 3: ["4", "5"]}) == {1: {"key_2": "value1"},
                                                                                    3: ["4", "5"]}


def test_iterdict_str_to_int_keys_with_nested_dicts():
    assert _iterdict_str_to_int_keys({"1": {"2": {"3": "value1"}}}) == {1: {2: {3: "value1"}}}


def test_iterdict_str_to_int_keys_with_nested_lists():
    assert _iterdict_str_to_int_keys({"1": {"2": ["3", {"4": "value1"}]}}) == {1: {2: ["3", {4: "value1"}]}}


@pytest.fixture()
def data():
    one_day = 60 * 24  # a day in minutes
    minutes = [0, one_day - 1, one_day - 1, one_day - 1, one_day * 2 - 1, one_day * 3, one_day * 4 + 2, one_day * 7]

    S = pd.Series("wake up", index=minutes)
    start = dt.datetime.strptime("16/04/2020 07:30", "%d/%m/%Y %H:%M")
    S.index = S.index.map(lambda e: start + dt.timedelta(minutes=e))
    S.index = S.index.round("min")  # minutes as the lowest unit of difference
    S[3] = "coffee"
    return S


def test_fit(data):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)

    assert pcm.is_datetime_ is True
    assert pcm.auto_time_scale is True
    assert pcm.n_zeros_ > 0
    assert [*pcm.alpha_groups.keys()] == ["coffee", "wake up"]
    assert len(pcm.alpha_groups["coffee"]) == 1
    assert len(pcm.alpha_groups["wake up"]) == 6  # one duplicate has been removed
    expected_data_details = {
        "t_start": "",
        "t_end": "",
        "deltaT": "",
        "nbOccs": {1: 6, 0: 1, -1: 7},
        "orgFreqs": {1: 6 / 7, 0: 1 / 7},
        "adjFreqs": {1: 6 * 1 / (3 * 7), 0: 1 * 1 / (3 * 7), '(': 1 / 3, ')': 1 / 3},
        "blck_delim": -2 * np.log2(1 / 3)
    }
    assert pcm.data_details.data_details["nbOccs"] == expected_data_details["nbOccs"]
    assert pcm.data_details.data_details["orgFreqs"] == expected_data_details["orgFreqs"]
    assert pcm.data_details.data_details["adjFreqs"] == expected_data_details["adjFreqs"]
    assert pcm.data_details.data_details["blck_delim"] == expected_data_details["blck_delim"]
    assert pcm.miners_ is not None


def test_discover(data):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)
    res_discover = pcm.discover()

    assert len(res_discover.columns) == 7
    assert res_discover["t0"].dtypes.name == "datetime64[ns]"
    assert res_discover["info"].dtypes.name == "object"
    assert res_discover["length_major"].dtypes.name == "int64"
    assert res_discover["period_major"].dtypes.name == "timedelta64[ns]"
    assert res_discover["cost"].dtypes.name == "float64"
    assert res_discover["type"].dtypes.name == "object"
    assert res_discover["E"].dtypes.name == "timedelta64[ns]"

    res_discover = pcm.discover(dE_sum=False)
    assert res_discover["E"].dtypes.name == "object"


def test_discover_chronological_order():
    pcm = PeriodicPatternMiner()
    data = fetch_health_app()
    pcm.fit(data[:100])
    res_discover_not_sorted = pcm.discover(chronological_order=False)
    assert res_discover_not_sorted["t0"].is_monotonic_increasing is False

    res_discover_sorted = pcm.discover(chronological_order=True)
    assert res_discover_sorted["t0"].is_monotonic_increasing is True


@pytest.fixture
def patterns_json():
    return {
        "is_datetime_": True,
        "n_zeros_": 10,
        "data_details": {
            "coffee": [
                158710854
            ],
            "wake up": [
                158702220,
                158710854,
                158719494,
                158728140,
                158736792,
                158762700
            ]
        },
        "patterns": [
            {
                "0": {
                    "p": 8643,
                    "r": 5,
                    "children": [
                        [
                            1,
                            0
                        ]
                    ],
                    "parent": None
                },
                "1": {
                    "event": 1,
                    "parent": 0
                },
                "next_id": 2,
                "t0": 158702220,
                "E": [
                    -9,
                    -3,
                    3,
                    9
                ]
            }
        ]
    }


def test_export_patterns(data, patterns_json):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)
    pcm.export_patterns()

    with patch("builtins.open", mock_open()) as mock_file:
        pcm.export_patterns()

        handle = mock_file()
        handle.write.assert_called_once_with(json.dumps(patterns_json))


# def test_reconstruct():
#
# def test_get_residuals():
#
#
# def test_import_patterns():
#
# def test_export_patterns():
