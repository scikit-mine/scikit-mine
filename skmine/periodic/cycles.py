"""Periodic pattern mining with a MDL criterion"""
# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from itertools import groupby
import numpy as np
import pandas as pd

from ..bitmaps import Bitmap
from ..utils import intersect2d, sliding_window_view
from ..base import BaseMiner, DiscovererMixin, MDLOptimizer

log = np.log2

INDEX_TYPES = (
    pd.DatetimeIndex,
    pd.RangeIndex,
    pd.Int64Index,
)

import warnings


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
    card = S.shape[0] if isinstance(S, np.ndarray) else 1
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
    tuple:
        lengths for (a, r, p, tau, E)
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


def compute_cycles_dyn(S, n_tot, max_length=100):
    """
    Parameters
    ----------
    S: pd.Index or np.array
        a Series of occurences
    n_tot: int
        total number of occurences in the original events
    """
    _, cut_points = get_table_dyn(S, n_tot, max_length)
    splits = _recover_splits_rec(cut_points, 0, len(S) - 1)

    cycles = list()
    covered = Bitmap()
    for length, g in groupby(splits, key=lambda e: e[1] - e[0]):
        g = list(g)
        if length >= 2:  # eq to length + 1 >= 3
            curr_cycles = np.vstack([S[s : e + 1] for s, e in g])  # COPY
            cycles.append(curr_cycles)
            for s, e in g:
                covered.update(range(s, e + 1))

    return list(reversed(cycles)), covered


def get_table_dyn(S: pd.Index, n_tot: int, max_length=100):
    """
    Parameters
    ----------
    S: pd.Index or np.ndarray
        a Series of occurences
    n_tot: int
        total number of occurences in the original events
    max_length: int, default=None
        maximum number of occurences for a cycle to cover,
        by default it will be set to :math:`\log_{2}\left(|S|\right)`
    """
    diffs = np.diff(S)
    triples = sliding_window_view(S, 3)
    diff_pairs = sliding_window_view(diffs, 2)
    dS = S.max() - S.min()

    score_one = residual_length(1, n_tot, dS)

    scores = sum(cycle_length(triples, diff_pairs, len(S), dS))
    change = scores > 3 * score_one
    scores[change] = 3 * score_one  # inplace replacement
    cut_points = np.array([-1] * len(scores), dtype=object)
    cut_points[~change] = None

    scores = dict(zip(((i, i + 2) for i in range(len(scores))), scores))
    cut_points = dict(zip(scores.keys(), cut_points))

    max_length = min([len(S), max_length])
    for k in range(4, max_length + 1):
        w = sliding_window_view(S, k)
        _diffs = sliding_window_view(diffs, k - 1)
        _s = sum(cycle_length(w, _diffs, len(S), dS))

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


def extract_triples(S, dS):
    """
    Extract cycles of length 3 given a list of occurences S
    Parameters
    ----------
    S: pd.Index
        input occurences

    dS
        difference between max event and min event, from original Series
    """
    l_max = log(dS + 1) - 2
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
            e = np.array([t[:, 0], np.array([occ] * t.shape[0]), t[:, 1]]).T
            assert np.issubdtype(e.dtype, np.number) and e.shape[1] == 3
            triples.append(e)
    if triples:
        return np.vstack(triples)
    return None


def merge_triples(triples, n_merge=10):
    """
    Parameters
    ----------
    triples: ndarray
        cycles of size 3 (i.e triples.shape[1] == 3)
    n_merge: int
        maximum number of merge operation to perform

    Returns
    -------
    list[np.ndarray]
        a list of cycles
    """
    if triples is None:
        return list()
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
        if (np.unique(merged) == np.unique(merged)).all():
            res.pop(idx - 1)
            break
    return list(reversed(res))  # inverse order of length


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


def _generate_candidates(S_a: pd.Index, n_event_tot: int, max_length: int = 100):
    if len(S_a) < 3:
        return list()
    S_a = S_a.sort_values()
    dS = S_a[-1] - S_a[0]
    cycles, covered = compute_cycles_dyn(S_a, n_event_tot, max_length)
    covered = Bitmap(covered)

    if len(S_a) - len(covered) > 3:  # add triples if necessary
        _all = Bitmap(range(len(S_a)))
        _S_a = S_a[_all - covered]
        triples = extract_triples(_S_a, dS)
        merged = merge_triples(triples)
        cycles.extend(merged)

    return list(sorted(cycles, key=lambda _: _.shape[1], reverse=True))


def evaluate(S, cands):
    """
    Evaluate candidates `cands`, given S

    Unlike the original implementation by Galbrun & Al.,
    if an occurence is present in more than one candidate cycle, we keep the cycle
    with the greatest length.

    Parameters
    ----------
    S: pd.Index
        Series of occurences for a specific event
    cands: list[np.ndarray]
        A list of candidate batches. Each batch contains candidates occurences of the same length,
        hence stored in a common `numpy.ndarray`.
        Batches are sorted by decreasing order of width,
        so that we consider larger candidates first.
    """
    res = list()  # list of pandas DataFrame
    covered = list()

    for cand_batch in cands:
        seen_occs = np.isin(cand_batch, covered).any(axis=1)
        cand_batch = cand_batch[~seen_occs]  # larger candidates already seen
        length = cand_batch.shape[1]
        E = np.diff(cand_batch, axis=1)
        period = np.floor(np.median(E, axis=1)).astype("int64")
        dE = (E.T - period).T
        df = pd.DataFrame(
            dict(start=cand_batch[:, 0], length=length, period=period, dE=dE.tolist())
        )
        res.append(df)
        covered.extend(np.unique(cand_batch))

    res = pd.concat(res, ignore_index=True)
    residual_pos = Bitmap(range(len(S))) - Bitmap(np.searchsorted(S, sorted(covered)))
    return res, S[residual_pos]


class PeriodicCycleMiner(BaseMiner, MDLOptimizer, DiscovererMixin):
    """
    Mining periodic cycles with a MDL Criterion

    PeriodicCycleMiner is an approach to mine periodic cycles from event logs
    while relying on a Minimum Description Length (MDL) criterion to evaluate
    candidate cycles. The goal here is to extract a set of cycles that characterizes
    the periodic structure present in the data

    A cycle is defined a 5-tuple of of the form
        .. math:: \\alpha, r, p, \\tau, E

    Where

    - :math:`\\alpha` is the `repeating event`
    - :math:`r` is the number of repetitions of the event, called the `cycle length`
    - :math:`p` is the inter-occurence distance, called the `cycle period`
    - :math:`\\tau` is the index of the first occurence, called the `cycle starting point`
    - :math:`E` is a list of :math:`r - 1` signed integer offsets, i.e `cycle shift corrections`


    Parameters
    ----------
    max_length: int, default=100
        maximum length for a candidate cycle, when running the dynamic programming heuristic
    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single event is attributed a job
        to discover potential cycles.
        Threads are preffered over processes.

    Examples
    --------
    >>> from skmine.periodic import PeriodicCycleMiner
    >>> S = pd.Series("ring_a_bell", [10, 20, 32, 40, 60, 79, 100, 240])
    >>> pcm = PeriodicCycleMiner().fit(S)
    >>> pcm.discover()
                   start  length  period
    ring_a_bell 0     40       4      20
                1     10       3      11

    References
    ----------
    .. [1]
        Galbrun, E & Cellier, P & Tatti, N & Termier, A & Crémilleux, B
        "Mining Periodic Pattern with a MDL Criterion"
    """

    def __init__(self, *, max_length=100, n_jobs=1):
        self.cycles_ = pd.DataFrame()
        self.residuals_ = dict()
        self.is_datetime_ = None
        self.n_zeros_ = 0
        self.is_fitted = lambda: self.is_datetime_ is not None  # TODO : this make pickle broken
        self.n_jobs = n_jobs
        self.max_length = max_length

    def fit(self, S):
        """fit PeriodicCycleMiner on data logs

        This generate new candidate cycles and evaluate them.
        Residual occurences are stored as an internal attribute,
        for later reconstruction (MDL is lossless)

        Parameters
        -------
        S: pd.Series
            logs, represented as a pandas Series
            This pandas Series must have an index of type in
            (pd.DatetimeIndex, pd.RangeIndex, pd.Int64Index)
        """
        if not isinstance(S, pd.Series):
            raise TypeError("S must be a pandas Series")

        if not isinstance(S.index, INDEX_TYPES):
            raise TypeError(f"S must have an index with a type amongst {INDEX_TYPES}")

        self.is_datetime_ = isinstance(S.index, pd.DatetimeIndex)

        if S.index.duplicated().any():
            warnings.warn(f"found duplicates in S, removing them")
            S = S.groupby(S.index).first()

        S = S.copy()
        S.index, self.n_zeros_ = _remove_zeros(S.index.astype("int64"))

        candidates = self.generate_candidates(S)

        gr = S.groupby(S.values).groups
        cycles, residuals = zip(
            *(evaluate(gr[event], cands) for event, cands in candidates.items())
        )

        c = dict(zip(candidates.keys(), cycles))
        cycles = pd.concat(c.values(), keys=c.keys())
        residuals = dict(zip(candidates.keys(), residuals))
        residuals = {**gr, **residuals}  # fill groups with no cands with all occurences
        self.cycles_, self.residuals_ = cycles, residuals

        return self

    evaluate = evaluate

    def generate_candidates(self, S):
        """
        Parameters
        ----------
        S: pd.Index or numpy.ndarray
            Series of occurences for a specific event

        Returns
        -------
        dict[object, list[np.ndarray]]
            A dict, where each key is an event and each value a list of batch of candidates.
            Batches are sorted in inverse order of width,
            so that we consider larger candidate cycles first.
        """
        n_event_tot = S.shape[0]
        alpha_groups = S.groupby(S.values)

        candidates = dict()
        for event, S_a in alpha_groups:
            cands = _generate_candidates(S_a.index, n_event_tot, self.max_length)
            if cands:
                candidates[event] = cands

        return candidates

    def discover(self):
        """Return cycles as a pandas DataFrame, with 3 columns,
        with a 2-level multi-index: the first level mapping events,
        and the second level being positional

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  =================================
                start       when the cycle starts
                length      number of occurences in the event
                period      inter-occurence delay
                ==========  =================================

        Example
        -------
        >>> from skmine.periodic import PeriodicCycleMiner
        >>> S = pd.Series("ring", [10, 20, 32, 40, 60, 79, 100, 240])
        >>> pcm = PeriodicCycleMiner().fit(S)
        >>> pcm.discover()
                start  length  period
        ring 0     40       4      20
             1     10       3      11

        """
        if not self.is_fitted():
            raise Exception(f"{type(self)} instance if not fitted")
        cycles = self.cycles_[["start", "length", "period"]].copy()
        cycles.loc[:, ["start", "period"]] = cycles[["start", "period"]] * (
            10 ** self.n_zeros_
        )

        if self.is_datetime_:
            cycles.loc[:, "start"] = cycles.start.astype("datetime64[ns]")
            cycles.loc[:, "period"] = cycles.period.astype("timedelta64[ns]")

        return cycles

    def reconstruct(self):
        """Reconstruct the original occurences from the current cycles
        Residuals will also be included, as the compression scheme is lossless

        Denoting as :math:`\sigma(E)` the sum of the shift corrections for a cycle
        :math:`C`, we have

        .. math::
            \Delta(C)=(r-1) p+\sigma(E)

        Returns
        -------
        pd.Series
            The reconstructed dataset

        Notes
        -----
        The index of the resulting pd.Series will not be sorted
        """
        cycles = self.cycles_[["start", "period", "dE"]]
        result = list()
        cycles_groups = cycles.groupby(level=0)
        for alpha, df in cycles_groups:
            l = list()
            for start, period, dE in df.values:
                occurences = _reconstruct(start, period, dE)
                l.extend(occurences)
            residuals = pd.Series(alpha, index=self.residuals_.get(alpha, list()))
            S = pd.concat([residuals, pd.Series(alpha, index=l)])
            #S.index = S.index.sort_values()
            result.append(S)

        for event in self.residuals_.keys() - cycles_groups.groups.keys():  # add unfrequent events
            result.append(pd.Series(event, index=self.residuals_[event]))

        S = pd.concat(result)
        S.index *= 10 ** self.n_zeros_
        if self.is_datetime_:
            S.index = S.index.astype("datetime64[ns]")

        # TODO : this is due to duplicate event in the cycles, handle this in .evaluate()
        S = S.groupby(S.index).first()

        return S
