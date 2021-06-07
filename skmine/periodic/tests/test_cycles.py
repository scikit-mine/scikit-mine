import datetime as dt

import numpy as np
import pandas as pd
import pytest

from .. import cycles
from ..cycles import (
    PeriodicCycleMiner,
    SingleEventPeriodicMiner,
    _generate_candidates_batch,
    _recover_splits_rec,
    compute_cycles_dyn,
    cycle_length,
    evaluate,
    extract_triples,
    generate_candidates,
    get_table_dyn,
    merge_triples,
    residual_length,
    sliding_window_view,
)


@pytest.fixture
def minutes():
    cuts = np.arange(3) * 400
    smalls = np.arange(0, 20, 2).reshape((10, 1))
    return (cuts + smalls).T.reshape(-1)


def index_to_datetime(index, scale="minutes"):
    now = dt.datetime(2021, 4, 20, 12, 15, 00)
    index = index.map(lambda e: dt.timedelta(minutes=e))  # TODO minutes as scale
    return pd.to_datetime(index + now)


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


@pytest.fixture
def triples():
    return np.array(
        [
            [0, 2, 4],
            [0, 2, 6],
            [0, 4, 6],
            [2, 4, 6],
            [400, 402, 404],
            [400, 402, 406],
            [400, 404, 406],
            [402, 404, 406],
        ]
    )


@pytest.mark.parametrize("k", [3, 5])
def test_window_view(minutes, k):
    w = sliding_window_view(minutes, k)
    assert w.shape == (len(minutes) - k + 1, k)
    np.testing.assert_array_equal(w[0], minutes[:k])
    np.testing.assert_array_equal(w[-1], minutes[-k:])


def test_cycle_length_triples(minutes):
    triples = sliding_window_view(minutes, 3)
    inter = sliding_window_view(np.diff(minutes), 2)
    delta_S = minutes[-1] - minutes[0]
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
    assert _recover_splits_rec(cut_points, 0, 7) == [(0, 3), (4, 7)]


def test_compute_cycles_dyn():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    occs, covered = compute_cycles_dyn(minutes, len(minutes))
    assert covered == set(range(len(minutes)))
    assert isinstance(occs, list)


def test_compute_cycles_dyn_different_split_sizes(monkeypatch):
    minutes = np.array([0, 2, 4, 6, 8, 10, 400, 402, 404, 406, 408, 410])

    monkeypatch.setattr(
        cycles, "_recover_splits_rec", lambda *args: [(0, 2), (3, 5), (6, 11)]
    )

    occs, covered = compute_cycles_dyn(minutes, len(minutes))
    assert covered == set(range(len(minutes)))
    assert all([isinstance(e, np.ndarray) for e in occs])
    assert [e.shape for e in occs] == [(1, 6), (2, 3)]


def test_extract_triples(triples):
    minutes = pd.Index(np.array([0, 2, 4, 6, 400, 402, 404, 406]))
    delta_S = minutes[-1] - minutes[0]
    l_max = np.log2(delta_S + 1) - 2
    t = extract_triples(minutes, l_max)
    assert t.ndim == 2
    np.testing.assert_array_equal(triples, t)


def test_merge_triples(triples):
    merged = merge_triples(triples)
    # original triples (size 3) should all have been merged into larger candidates
    assert len(merged) == 1
    np.testing.assert_array_equal(np.unique(merged[0]), np.unique(triples))


def test_generate_candidate_batch():
    minutes = pd.Index([0, 20, 31, 40, 60, 240, 400, 420, 440, 460])
    cands = _generate_candidates_batch(minutes, len(minutes))
    widths = [_.shape[1] for _ in cands]
    assert all(x > y for x, y in zip(widths, widths[1:]))  # monotonic but decreasing


def test_generate_candidates():
    minutes = np.array([0, 2, 4, 6, 8, 10, 400, 402, 404, 406, 408, 410])
    cycles = generate_candidates(minutes, len(minutes))
    assert cycles.length.sum() == len(minutes)
    _types = np.unique(cycles.dtypes)
    assert set(cycles.columns.to_list()).issuperset({"start", "length", "period", "dE"})
    assert np.issubdtype(_types[0], np.number)
    assert cycles.length.tolist() == [6, 6]  # [6, 3, 3] ?
    assert (cycles.period == 2).all()
    assert cycles.index.is_monotonic_increasing


def test_evaluate_overlapping_candidates():
    minutes = np.array([0, 2, 4, 6, 8, 10, 400, 402, 404, 406, 408, 410])
    cands = generate_candidates(minutes, len(minutes))
    cycles = evaluate(cands, k=2)
    _types = np.unique(cycles.dtypes)
    assert set(cycles.columns.to_list()).issuperset({"start", "length", "period", "dE"})
    assert np.issubdtype(_types[0], np.number)
    assert cycles.length.tolist() == [6, 6]
    assert (cycles.period == 2).all()


def test_fit():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    S = pd.Series("alpha", index=minutes)
    S.index = index_to_datetime(S.index)
    S.index = pd.to_datetime(S.index)

    pcm = PeriodicCycleMiner()
    pcm.fit(S)

    assert len(pcm.miners_) == 1
    assert sum(map(len, (m.cycles_ for m in pcm.miners_.values()))) == 2
    assert all(
        "dE" in c.columns for c in map(lambda m: m.cycles_, pcm.miners_.values())
    )


def test_discover():
    minutes = np.array([0, 2, 4, 6, 400, 402, 404, 406])

    S = pd.Series("alpha", index=minutes)
    S.index = index_to_datetime(S.index)
    S.index = pd.to_datetime(S.index)
    pcm = PeriodicCycleMiner()
    cycles = pcm.fit_discover(S)
    assert (cycles.dtypes != "object").all()  # only output structured data


@pytest.mark.parametrize("is_datetime", (True, False))
def test_reconstruct(is_datetime):
    minutes = np.array([0, 2, 4, 6, 400, 400, 402, 404, 406])

    S = pd.Series("alpha", index=minutes)
    # add infrequent item to be included in reconstructed
    S = S.append(pd.Series("beta", index=[10]))
    S.sort_index(inplace=True)

    if is_datetime:
        S.index = index_to_datetime(S.index)

    pcm = PeriodicCycleMiner(keep_residuals=True).fit(S)
    assert pcm.is_datetime_ == is_datetime
    reconstructed = pcm.reconstruct()
    pd.testing.assert_index_equal(reconstructed.index, S.index.drop_duplicates())


def test_fit_triples_and_residuals():
    minutes = np.array([0, 20, 31, 40, 60, 240, 400, 420, 431, 440, 460, 781])

    S = pd.Series("alpha", index=minutes)

    pcm = PeriodicCycleMiner(keep_residuals=True).fit(S)
    # pd.testing.assert_index_equal(pcm.residuals_["alpha"], pd.Int64Index([240, 781]))

    rec_minutes = pcm.reconstruct()

    pd.testing.assert_series_equal(S, rec_minutes)


def test_duplicates():
    S = pd.Series("alpha", index=[20, 20, 40, 50])
    with pytest.warns(UserWarning):
        PeriodicCycleMiner().fit(S)


def test_small_datetime():
    minutes = [10, 20, 32, 40, 60, 79, 100, 240]
    # just check it does not break
    S = pd.Series("ring_a_bell", index=minutes)
    S.index = index_to_datetime(S.index)
    pcm = PeriodicCycleMiner().fit(S)
    cycles = pcm.discover(shifts=True)
    assert "dE" in cycles.columns


@pytest.mark.parametrize(
    "_input,raise_warning",
    [([20, 31, 40, 60, 240], False), ([20, 35, 40, 240], True), ([20, 40, 50], False)],
)
def test_candidates(_input, raise_warning):
    minutes = np.array(_input)

    with pytest.warns(None, match="candidate") as record:
        SingleEventPeriodicMiner().fit(minutes)

    assert len(record) == int(raise_warning)


@pytest.mark.parametrize("keep_residuals", [True, False])
def test_get_residuals(keep_residuals):
    minutes = np.array([0, 20, 31, 40, 60, 154, 240, 270, 300, 330, 358])

    S = pd.Series("alpha", index=minutes)

    pcm = PeriodicCycleMiner(keep_residuals=keep_residuals).fit(S)
    residuals = pcm.get_residuals()
    # assert isinstance(residuals.index, pd.DatetimeIndex)
    np.testing.assert_array_equal(residuals.index, [154] * int(keep_residuals))
