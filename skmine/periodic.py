import numpy as np

import pandas as pd
from .bitmaps import Bitmap
from .utils import intersect2d

from skmine.base import BaseMiner, DiscovererMixin, MDLOptimizer
from joblib import Parallel, delayed

log = np.log2


def window_stack(x, width=3):
    """
    Returns
        np.array of shape (x.shape[0] - width + stepsize, width)
    """
    n = x.shape[0]
    subs = [x[i : 1 + n + i - width] for i in range(0, width)]
    return np.vstack(subs).T


def residual_length(S, n_event_tot, dS):
    """
    compute L(o) = L(t) + L(a) for all (a, t) in S
    i.e the length from a block of residual events

    Parameters
    ----------
    S: np.ndarray of shape or scalar
        array containing indices for events to consider

    n_event_tot: int
        number of events in the original events

    dS: int
        max - min from original events
    """
    if isinstance(S, np.ndarray):
        card = S.shape[0]
    else:  # single value as scalar
        card = 1
    return log(dS + 1) - log(card / float(n_event_tot))


def cycle_length(S, inter, n_event_tot, dS):
    """
    Parameters
    ----------
    S : np.array of type int64
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
    r = S.shape[1]
    assert inter.shape[1] == r - 1  # check inter occurences compliant with events
    p = np.median(inter, axis=1)
    E = inter - p.reshape((-1, 1))
    dE = E.sum(axis=1)
    S_size = len(S) + r - 1

    L_a = -log(S_size / n_event_tot)  # FIXME
    L_r = log(S_size)
    L_p = log(np.floor((dS - dE) / (r - 1)))
    L_tau = log(dS - dE - (r - 1) * p + 1)
    L_E = 2 * E.shape[1] + np.abs(E).sum(axis=1)

    return L_a, L_r, L_p, L_tau, L_E


def compute_cycles_dyn(S, n_tot):
    """
    Parameters
    ----------
    S: pd.Index or np.array
        a Series of occurences
    n_tot: int
        total number of occurences in the original events
    """
    _, cut_points = get_table_dyn(S, n_tot)
    splits = _recover_splits_rec(cut_points, 0, len(S) - 1)

    cycles = list()
    covered = set()
    for start, end in splits:
        length = end - start + 1
        if length >= 3:
            cov = S[start : start + length]
            E = np.diff(cov)
            period = np.floor(np.median(E)).astype("int64")
            dE = E - period
            # TODO : compute score ?
            cycles.append([cov[0], length, period, dE])
            covered.update(range(start, end + 1))

    cycles = pd.DataFrame(cycles, columns=["start", "length", "period", "dE"])
    return cycles, covered


def get_table_dyn(S: pd.Index, n_tot: int):
    """
    Parameters
    ----------
    S: pd.Index or np.ndarray
        a Series of occurences
    n_tot: int
        total number of occurences in the original events
    """
    S = S.astype("int64")
    diffs = np.diff(S)
    triples = window_stack(S, width=3)
    diff_pairs = window_stack(diffs, width=2)
    delta_S = S.max() - S.min()

    score_one = residual_length(1, n_tot, delta_S)

    scores = sum(cycle_length(triples, diff_pairs, len(S), delta_S))
    change = scores > 3 * score_one
    scores[change] = 3 * score_one  # inplace replacement
    cut_points = np.array([-1] * len(scores), dtype=object)
    cut_points[~change] = None

    scores = dict(zip(((i, i + 2) for i in range(len(scores))), scores))
    cut_points = dict(zip(scores.keys(), cut_points))

    for k in range(4, len(S) + 1):
        w = window_stack(S, width=k)
        _diffs = window_stack(diffs, width=k - 1)
        _s = sum(cycle_length(w, _diffs, len(S), delta_S))

        for ia, best_score in enumerate(_s):
            cut_point = None
            iz = ia + k - 1
            for im in range(ia, iz):
                if im - ia + 1 < 3:
                    score_left = score_one * (im - ia + 1)
                else:
                    score_left = scores[(ia, im)]
                if iz - im < 3:
                    score_right = score_one * (iz - im)
                else:
                    score_right = scores[(im + 1, iz)]

                if score_left + score_right < best_score:
                    best_score = score_left + score_right
                    cut_point = im
            scores[(ia, iz)] = best_score
            cut_points[(ia, iz)] = cut_point

    return scores, cut_points


def extract_triples(S, delta_S):
    """
    Extract cycles of length 3 given a list of occurences S
    Parameters
    ----------
    S: pd.Index
        input occurences

    delta_S
        difference between max event and min event, from original Series
    """
    if not S.is_monotonic_increasing:
        S = S.sort_values()
    l_max = log(delta_S + 1) - 2
    triples = list()

    for idx, occ in enumerate(S[1:-1], 1):
        righties = S[idx + 1 :]
        lefties = S[:idx]
        righties_diffs = righties - occ
        lefties_diffs = lefties - occ
        grid = np.array(np.meshgrid(lefties_diffs, righties_diffs)).T.reshape(-1, 2)
        # keep = (np.abs(grid[:, 1]) - np.abs(grid[:, 0])) <= l
        keep = np.abs(grid[:, 0] - grid[:, 1]) < l_max
        t = occ + grid[keep]
        if t.size != 0:
            e = np.array([t[:, 0], np.array([occ] * t.shape[1]), t[:, 1]]).T
            triples.append(e)
            # covered.update(np.searchsorted(S.values, e).reshape(-1))

    triples = np.vstack(triples)
    return triples


def merge_triples(triples, n_merge=10):
    """
    Parameters
    ----------
    triples: ndarray
        cycles of size 3 (i.e triples.shape[1] == 3)
    n_merge: int
        maximum number of merge operation to perform
    """
    res = [triples]
    for idx in range(1, n_merge + 1):
        prev = res[idx - 1]
        lefties = prev[:, :2]
        righties = prev[:, -2:]
        _, left_idx, right_idx = intersect2d(lefties, righties, return_indices=True)
        if left_idx.size == 0 or right_idx.size == 0:
            break
        merged = np.hstack([prev[right_idx], prev[left_idx, 2:]])
        res.append(merged)
        to_delete = np.union1d(left_idx, right_idx)
        res[idx - 1] = np.delete(res[idx - 1], to_delete, axis=0)
    return res


def _recover_splits_rec(cut_points, ia, iz):
    if (ia, iz) in cut_points:
        if cut_points[(ia, iz)] is None:
            return [(ia, iz)]
        im = cut_points[(ia, iz)]
        if im >= 0:
            return _recover_splits_rec(cut_points, ia, im) + _recover_splits_rec(
                cut_points, im + 1, iz
            )
    return []


def _remove_zeros(numbers: pd.Series):
    n = 0
    while (numbers % 10 == 0).all():
        numbers //= 10
        n += 1
    return numbers, n


def _reconstruct(start, period, dE):
    """
    Reconstruct occurences,
    starting from `start`, and
    correcting `period` with a delta for all deltas in `dE`,
    `len(dE)` occurences are reconstructed

    Parameters
    ----------
    start: int or datetime
        starting point for the event
    period: int or timedelta
        period between two occurences
    d_E: np.array of [int|timedelta]
        inters occurences deltas
    """
    occurences = [start]
    current = start
    for d_e in dE:
        e = current + period + d_e
        occurences.append(e)
        current = e
    return occurences


# TODO : inherit MDLOptimizer
class PeriodicCycleMiner(BaseMiner, DiscovererMixin):
    def __init__(self):
        self.cycles_ = pd.DataFrame()
        self.is_datetime_ = None
        self.n_zeros_ = 0

    def fit(self, S):
        if not isinstance(S, pd.Series):
            raise TypeError("S must be a pandas Series")

        if not isinstance(S.index, (pd.RangeIndex, pd.DatetimeIndex)):
            raise TypeError("S must have an index of type RangeIndex of DatetimeIndex")

        if not S.index.is_monotonic:
            raise TypeError("S must have a monotonic index")

        self.is_datetime_ = isinstance(S.index, pd.DatetimeIndex)

        S = S.copy()
        S.index, self.n_zeros_ = _remove_zeros(S.index.astype("int64"))

        self.cycles_ = self.generate_candidates(S)
        # TODO fitler candidates
        return self

    def generate_candidates(self, S):
        n_event_tot = S.shape[0]
        alpha_groups = S.groupby(S.values)

        cycles, covered = zip(
            *Parallel()(
                delayed(compute_cycles_dyn)(S_a.index, n_event_tot)
                for _, S_a in alpha_groups
            )
        )
        # covered = reduce(set.union, covered)
        cycles = dict(zip(alpha_groups.groups.keys(), cycles))
        return pd.concat(cycles.values(), keys=cycles.keys())  # return multiindex

    def discover(self):
        cycles = self.cycles_[["start", "length", "period"]].copy()
        cycles.loc[:, ["start", "period"]] = cycles[["start", "period"]] * (
            10 ** self.n_zeros_
        )

        if self.is_datetime_:
            cycles.loc[:, "start"] = cycles.start.astype("datetime64[ns]")
            cycles.loc[:, "period"] = cycles.period.astype("timedelta64[ns]")

        return cycles

    def reconstruct(self):
        cycles = self.cycles_[["start", "period", "dE"]]
        result = list()
        for alpha, df in cycles.groupby(level=0):
            l = list()
            for start, period, dE in df.values:
                occurences = _reconstruct(start, period, dE)
                l.extend(occurences)
            S = pd.Series(alpha, index=l)
            result.append(S)
        S = pd.concat(result)
        S.index *= 10 ** self.n_zeros_
        if self.is_datetime_:
            S.index = S.index.astype("datetime64[ns]")

        return S
