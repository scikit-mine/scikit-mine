import pytest
import numpy as np

from .periodic import get_cycles_dyn, window_stack, residual_length, cycle_length


@pytest.fixture
def minutes():
    cuts = np.arange(3) * 400
    smalls = np.arange(0, 20, 2).reshape((10, 1))
    return (cuts + smalls).T.reshape(-1)


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
