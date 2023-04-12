import datetime as dt
import json
from unittest.mock import patch, mock_open

import graphviz
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skmine.datasets import fetch_health_app
from skmine.periodic.cycles import _remove_zeros, _iterdict_str_to_int_keys, PeriodicPatternMiner


def test_remove_zeros():
    numbers = pd.Index([1587022200000000000, 1587108540000000000, 1587194940000000000,
                        1587281400000000000, 1587367920000000000, 1587627000000000000], dtype='int64')
    expected_output = (pd.Index([158702220, 158710854, 158719494, 158728140, 158736792, 158762700], dtype='int64'),
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
    assert [*pcm.alpha_groups_.keys()] == ["coffee", "wake up"]
    assert len(pcm.alpha_groups_["coffee"]) == 1
    assert len(pcm.alpha_groups_["wake up"]) == 6  # one duplicate has been removed
    expected_data_details = {
        "t_start": "",
        "t_end": "",
        "deltaT": "",
        "nbOccs": {1: 6, 0: 1, -1: 7},
        "orgFreqs": {1: 6 / 7, 0: 1 / 7},
        "adjFreqs": {1: 6 * 1 / (3 * 7), 0: 1 * 1 / (3 * 7), '(': 1 / 3, ')': 1 / 3},
        "blck_delim": -2 * np.log2(1 / 3)
    }
    assert pcm.data_details_.data_details["nbOccs"] == expected_data_details["nbOccs"]
    assert pcm.data_details_.data_details["orgFreqs"] == expected_data_details["orgFreqs"]
    assert pcm.data_details_.data_details["adjFreqs"] == expected_data_details["adjFreqs"]
    assert pcm.data_details_.data_details["blck_delim"] == expected_data_details["blck_delim"]
    assert pcm.miners_ is not None


def test_discover(data):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)
    res_transform = pcm.transform(data)

    assert len(res_transform.columns) == 5
    assert res_transform["t0"].dtypes.name == "datetime64[ns]"
    assert res_transform["pattern"].dtypes.name == "object"
    assert res_transform["repetition_major"].dtypes.name == "int64"
    assert res_transform["period_major"].dtypes.name == "timedelta64[ns]"
    assert res_transform["sum_E"].dtypes.name == "timedelta64[ns]"

    res_transform = pcm.transform(data, dE_sum=False)
    assert res_transform["E"].dtypes.name == "object"


def test_discover_chronological_order():
    pcm = PeriodicPatternMiner()
    data = fetch_health_app()
    pcm.fit(data[:100])
    res_transform_not_sorted = pcm.transform(data, chronological_order=False)
    assert res_transform_not_sorted["t0"].is_monotonic_increasing is False

    res_transform_sorted = pcm.transform(data, chronological_order=True)
    assert res_transform_sorted["t0"].is_monotonic_increasing is True


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


def test_import_patterns(patterns_json):
    pcm = PeriodicPatternMiner()


    mock_file = mock_open(read_data=json.dumps(patterns_json))
    with patch("builtins.open", mock_file):
        pcm.import_patterns()

        mock_file.assert_called_once_with("patterns.json", "r")

    assert pcm.n_zeros_ == patterns_json["n_zeros_"]
    assert pcm.is_datetime_ == patterns_json["is_datetime_"]
    assert pcm.data_details_.data_details == {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 1, -1: 7},
        "orgFreqs": {1: 6 / 7, 0: 1 / 7},
        "adjFreqs": {1: 6 * 1 / (3 * 7), 0: 1 * 1 / (3 * 7), '(': 1 / 3, ')': 1 / 3},
        "blck_delim": -2 * np.log2(1 / 3)
    }
    assert len(pcm.miners_.patterns) == len(patterns_json["patterns"])
    assert list(pcm.miners_.patterns[0][0].nodes.keys()) == [0, 1]
    assert pcm.miners_.patterns[0][0].nodes[0] == patterns_json["patterns"][0]["0"]
    assert pcm.miners_.patterns[0][0].nodes[1] == patterns_json["patterns"][0]["1"]
    assert pcm.miners_.patterns[0][1] == patterns_json["patterns"][0]["t0"]
    assert pcm.miners_.patterns[0][2] == patterns_json["patterns"][0]["E"]


def test_import_export_patterns(data):
    pcm1 = PeriodicPatternMiner()
    pcm1.fit(data)
    dummy_var = 17
    res1 = pcm1.transform(dummy_var)
    pcm1.export_patterns()
    pcm2 = PeriodicPatternMiner()
    pcm2.import_patterns()
    res2 = pcm2.transform(dummy_var)

    assert_frame_equal(res1, res2)


@pytest.fixture
def expected_reconstruct():
    expected_reconstruct = pd.DataFrame({
        'time': ['2020-04-16 07:30:00', '2020-04-17 07:29:00', '2020-04-18 07:29:00', '2020-04-19 07:30:00',
                 '2020-04-20 07:32:00'],
        'event': ['wake up', 'wake up', 'wake up', 'wake up', 'wake up']
    })
    expected_reconstruct['time'] = pd.to_datetime(expected_reconstruct['time'])
    return expected_reconstruct


def test_reconstruct(data, expected_reconstruct):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)

    reconstruct = pcm.reconstruct()
    assert_frame_equal(reconstruct, expected_reconstruct)


def test_reconstruct_order():
    pcm = PeriodicPatternMiner()
    data = fetch_health_app()
    pcm.fit(data[:100])

    reconstruct_time = pcm.reconstruct()
    assert reconstruct_time["time"].is_monotonic_increasing is True

    reconstruct_event = pcm.reconstruct(sort="event")
    assert reconstruct_event["time"].is_monotonic_increasing is False
    assert reconstruct_event["event"].is_monotonic_increasing is True
    assert len(reconstruct_time) == len(reconstruct_event)

    reconstruct_construction = pcm.reconstruct(sort="construction_order")
    assert reconstruct_construction["time"].is_monotonic_increasing is False
    assert reconstruct_construction["event"].is_monotonic_increasing is False
    # by default, construction_order keeps duplicates to better understand the reconstruction
    assert len(reconstruct_construction) >= len(reconstruct_time)
    assert len(reconstruct_construction[reconstruct_construction.duplicated()]) == 1


def test_get_residuals(data, expected_reconstruct):
    pcm = PeriodicPatternMiner()
    pcm.fit(data)

    # Getting the complementary data with expected_reconstruct will give us the expected residuals
    data = pd.DataFrame({"time": data.index, "event": data.values})
    expected_residuals = pd.merge(data, expected_reconstruct, how='outer', indicator=True)
    expected_residuals = expected_residuals.loc[expected_residuals['_merge'] == 'left_only']
    expected_residuals = expected_residuals.drop('_merge', axis=1)

    residuals = pcm.get_residuals()
    assert_frame_equal(residuals, expected_residuals.reset_index(drop=True))


def test_get_residuals_health_app(data, expected_reconstruct):
    pcm = PeriodicPatternMiner()
    data = fetch_health_app()
    pcm.fit(data[:100])

    residuals_not_sorted = pcm.get_residuals()
    assert residuals_not_sorted["event"].is_monotonic_increasing is False

    residuals_event_sorted = pcm.get_residuals(sort="event")

    assert residuals_event_sorted["event"].is_monotonic_increasing is True


def test_draw_pattern(data):
    pcm = PeriodicPatternMiner().fit(data)
    res = pcm.transform(data)
    graph = pcm.draw_pattern(0)
    assert 0 in res.index
    assert type(graph) == graphviz.graphs.Digraph
    assert graph.source == 'digraph {\n\t0 [label="𝜏=2020-04-16 07:30:00\np=1 day, 0:00:30\nr=5" shape=box]\n\t1 [label="wake up"]\n\t0 -> 1 [dir=none]\n}\n'
