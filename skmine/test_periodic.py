import pytest
import numpy as np
import pandas as pd
from collections import Counter
import datetime as dt

from .periodic import (
    window_stack,
    residual_length,
    cycle_length,
    get_table_dyn,
    recover_splits_rec,
    compute_cycles_dyn,
    PeriodicCycleMiner,
)


@pytest.fixture
def minutes():
    cuts = np.arange(3) * 400
    smalls = np.arange(0, 20, 2).reshape((10, 1))
    return (cuts + smalls).T.reshape(-1)


@pytest.fixture
def cut_points():
    return {
        (2, 7): 3,
        (2, 5): 2,
        (1, 3): None,
        (4, 6): None,
        (1, 7): 3,
        (1, 6): 3,
        (0, 6): 3,
        (5, 7): None,
        (4, 7): None,
        (1, 4): 3,
        (2, 4): -1,
        (1, 5): 3,
        (2, 6): 3,
        (0, 5): 3,
        (0, 7): 3,
        (3, 6): 3,
        (0, 4): 3,
        (3, 7): 3,
        (0, 3): None,
        (0, 2): None,
        (3, 5): -1,
    }


@pytest.mark.parametrize("k", [3, 5])
def test_window_stack(minutes, k):
    w = window_stack(minutes, width=k, stepsize=1)
    assert w.shape == (len(minutes) - k + 1, k)
    np.testing.assert_array_equal(w[0], minutes[:k])
    np.testing.assert_array_equal(w[-1], minutes[-k:])


def test_cycle_length_triples(minutes):
    triples = window_stack(minutes)
    inter = window_stack(np.diff(minutes), width=2)
    delta_S = delta_S = minutes[-1] - minutes[0]
    L_a, L_r, L_p, L_tau, L_E = cycle_length(triples, inter, len(minutes), delta_S)

    # TODO : test L_a
    assert L_r == pytest.approx(4.9069, rel=1e-4)
    assert len(np.unique(L_p)) == 1
    assert np.unique(L_p)[0] == pytest.approx(8.6759, rel=1e-4)
    np.testing.assert_array_almost_equal(
        np.unique(L_tau), np.array([8.7649, 9.6706]), decimal=4
    )
    assert L_tau.mean() == pytest.approx(9.5413, rel=1e-4)

    np.testing.assert_array_almost_equal(
        np.unique(L_E), np.array([4.0, 384.0]), decimal=2
    )
    assert L_E.mean() == pytest.approx(58.2857)


@pytest.mark.parametrize(
    "idx,length", ([slice(0, 10), 11.2627], [2, 14.5846], [slice(0, 30), 9.6777])
)
def test_residual_length(minutes, idx, length):
    # np.log2(delta_S + 1) - np.log2(len(idx) / 30.)
    S_a = minutes[idx]
    delta_S = minutes[-1] - minutes[0]
    r = residual_length(S_a, len(minutes), delta_S)
    assert r == pytest.approx(length, rel=1e-4)


def test_get_table_dyn(cut_points):
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])
    scores, cut_points = get_table_dyn(minutes, len(minutes))
    expected_len = ((len(minutes) - 1) * (len(minutes) - 2)) / 2
    assert len(scores) == len(cut_points) == expected_len
    assert cut_points == cut_points
    assert np.mean(list(scores.values())) == pytest.approx(37.3237, rel=1e-4)


def test_recover_split_rec(cut_points):
    assert recover_splits_rec(cut_points, 0, 7) == [(0, 3), (4, 7)]


def test_compute_cycles_dyn():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    cycles, covered = compute_cycles_dyn(minutes, len(minutes))
    assert covered == set(range(len(minutes)))
    assert "start" in cycles.columns


def test_fit():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    S = pd.Series("alpha", index=minutes)
    S.index = S.index.map(lambda e: dt.datetime.now() + dt.timedelta(minutes=e))
    S.index = pd.to_datetime(S.index)
    pcm = PeriodicCycleMiner()
    pcm.fit(S)

    assert pcm.cycles_.index.to_series().nunique() == 2
    assert "inters" in pcm.cycles_.columns


def test_discover():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    S = pd.Series("alpha", index=minutes)
    S.index = S.index.map(lambda e: dt.datetime.now() + dt.timedelta(minutes=e))
    S.index = pd.to_datetime(S.index)
    pcm = PeriodicCycleMiner()
    pcm.fit(S)
    cycles = pcm.discover()
    assert (cycles.dtypes != "object").all()  # only output structured data
