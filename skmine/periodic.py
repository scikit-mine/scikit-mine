import numpy as np

import pandas as pd

from skmine.base import MDLOptimizer
from skmine.base import BaseMiner

log = np.log2


def window_stack(x, stepsize=1, width=3):
    """
    Returns
        np.array of shape (x.shape[0] - width + stepsize, width)
    """
    n = x.shape[0]
    subs = [x[i : 1 + n + i - width : stepsize] for i in range(0, width)]
    return np.vstack(subs).T


def residual_length(S_alpha, n_event_tot, delta_S):
    """
    compute L(o) = L(t) + L(a) for all (a, t) in S_alpha
    i.e the length from a block of residual events

    Parameters
    ----------
    S_alpha: np.narray of shape (|a|, ) or scalar
        array containing indices for events to consider

    n_event_tot: int
        number of events in the original events

    delta_S: int
        max - min from original events
    """
    if isinstance(S_alpha, np.ndarray):
        card = S_alpha.shape[0]
    else:  # single value as scalar
        card = 1
    return log(delta_S + 1) - log(card / float(n_event_tot))


def cycle_length(S_alpha, inter, n_event_tot, dS):
    """
    Parameters
    ----------
    S_alpha : np.array of type int64
        a collection of cycles, all having the same length : r
        The width of S is then r

    inter: np.array of type int64
        a collection of inter occurences, all having the same length: r - 1

    n_event_tot: int
        number of events in the original events

    dS: int
        max - min from original events

    Returns
    -------
    tuple()
    """
    r = S_alpha.shape[1]
    assert inter.shape[1] == r - 1  # check inter occurences compliant with events
    p = np.median(inter, axis=1)
    E = inter - p.reshape((-1, 1))
    dE = E.sum(axis=1)
    S_alpha_size = len(S_alpha) + r - 1

    L_a = -log(S_alpha_size / n_event_tot)  # FIXME
    L_r = log(S_alpha_size)
    L_p = log(np.floor((dS - dE) / (r - 1)))
    L_tau = log(dS - dE - (r - 1) * p + 1)
    L_E = 2 * E.shape[1] + np.abs(E).sum(axis=1)

    return L_a, L_r, L_p, L_tau, L_E


def get_table_dyn(S_a: pd.Series):
    pass


def get_cycles_dyn(S_a: pd.Series):
    """
    Parameters
    ----------
    S_a: pd.Series
        Series corresponding to a single event a

    Return
    ------
    pd.DataFrame
        DataFrame of cycles
    """

    scores, cut_points = get_table_dyn(S_a)
    return pd.DataFrame()


# TODO : inherit MDLOptimizer
class PeriodicCycleMiner(BaseMiner):
    def fit(self, S):
        if not isinstance(S, pd.Series) or not isinstance(S.index, pd.DatetimeIndex):
            raise TypeError("S must be a Series with a datetime index")

        alpha_groups = S.groupby(S.values)
        alpha_sizes = alpha_groups.apply(len)

        # TODO
