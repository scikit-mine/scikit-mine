import pandas as pd
import pytest

from skmine.periodic.data_sequence import DataSequence

seqs = {
    "wake up": pd.Int64Index([158702220, 158710854, 158719494, 158728140, 158736792, 158762700], dtype='int64'),
    "coffee": pd.Int64Index([158719499, 158702225], dtype='int64')
}


def test_data_sequence():
    ds = DataSequence(seqs)
    expected_list_ev = ["coffee", "wake up"]  # lexicographic order
    assert expected_list_ev == ds.list_ev

    expected_map_ev_num = {
        "coffee": 0,
        "wake up": 1
    }
    assert expected_map_ev_num == ds.map_ev_num

    expected_evStarts = {
        1: 158702220,
        0: 158702225
    }
    assert expected_evStarts == ds.evStarts

    expected_evEnds = {
        1: 158762700,
        0: 158719499
    }
    assert expected_evEnds == ds.evEnds

    expected_seqd = {
        0: [158719499, 158702225],
        1: [158702220, 158710854, 158719494, 158728140, 158736792, 158762700]
    }
    assert set(ds.seqd.keys()) == {0, 1}
    for ev_nb in expected_seqd.keys():
        assert (expected_seqd[ev_nb] == ds.seqd[ev_nb]).all()

    expected_seql = [
        (158719499, 0),
        (158702225, 0),
        (158702220, 1),
        (158710854, 1),
        (158719494, 1),
        (158728140, 1),
        (158736792, 1),
        (158762700, 1)
    ]
    assert set(expected_seql) == set(ds.seql)

    expected_data_details = {
        "t_start": 158702220,
        "t_end": 158762700,
        "deltaT": 60480,
        "nbOccs": {1: 6, 0: 2, -1: 8},
        "orgFreqs": {1: 0.75, 0: 0.25},
        "adjFreqs": {},  # FIXME
        "blck_delim": None  # FIXME
    }
    assert expected_data_details.keys() == ds.data_details.keys()
    assert expected_data_details["t_start"] == ds.data_details["t_start"]
    assert expected_data_details["t_end"] == ds.data_details["t_end"]
    assert expected_data_details["deltaT"] == ds.data_details["deltaT"]
    assert expected_data_details["nbOccs"] == ds.data_details["nbOccs"]
    assert expected_data_details["orgFreqs"] == ds.data_details["orgFreqs"]
