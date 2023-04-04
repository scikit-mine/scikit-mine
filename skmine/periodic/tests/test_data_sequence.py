import numpy as np
import pandas as pd

from skmine.periodic.data_sequence import DataSequence

seqs = {
    "wake up": pd.Index([158702220, 158710854, 158719494, 158728140, 158736792, 158762700], dtype='int64'),
    "coffee": pd.Index([158719499, 158702225], dtype='int64')
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
        "adjFreqs": {1: 6*1/(3*8), 0: 2*1/(3*8), '(': 1/3, ')': 1/3},
        "blck_delim": -2*np.log2(1/3)
    }
    assert expected_data_details == ds.data_details


def test_data_sequence_seq_list():
    seqs_list = [(158702220, "wake up"), (158710854, "wake up"), (158719494, "wake up"), (158728140, "wake up"),
                 (158736792, "wake up"), (158762700, "wake up"), (158719499, "coffee"), (158702225, "coffee")]

    ds = DataSequence(seqs_list)
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
        assert (expected_seqd[ev_nb] == ds.seqd[ev_nb])

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
        "adjFreqs": {1: 6 * 1 / (3 * 8), 0: 2 * 1 / (3 * 8), '(': 1 / 3, ')': 1 / 3},
        "blck_delim": -2 * np.log2(1 / 3)
    }
    assert expected_data_details == ds.data_details


def test_getInfoStr():
    ds = DataSequence(seqs)
    expected_output = "-- Data Sequence |A|=2 |O|=8 dT=60480 (158702220 to 158762700)" \
                      "\n\tcoffee [0] (|O|=2 f=0.250 dT=17274)" \
                      "\n\twake up [1] (|O|=6 f=0.750 dT=60480)"
    assert expected_output == ds.getInfoStr()


def test_getInfoStr_empty():
    ds = DataSequence([])
    assert "-- Empty Data Sequence" == ds.getInfoStr()


def test_getEvents():
    ds = DataSequence(seqs)
    assert ds.getEvents() == ["coffee", "wake up"]


def test_getNumToEv():
    ds = DataSequence(seqs)
    expected_output = {
        0: "coffee",
        1: "wake up"
    }
    assert expected_output == ds.getNumToEv()
