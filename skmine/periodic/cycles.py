"""Periodic pattern mining with a MDL criterion"""
# Authors: Rémi Adon <remi.adon@gmail.com>
#          Esther Galbrun <esther.galbrun@inria.fr>
#
# License: BSD 3 clause

import warnings
from itertools import groupby

import numpy as np
import pandas as pd

from pyroaring import BitMap as Bitmap

from ..base import BaseMiner, MDLOptimizer
from ..utils import intersect2d, sliding_window_view

log = np.log2

INDEX_TYPES = (
    pd.DatetimeIndex,
    pd.RangeIndex,
    pd.Int64Index,
)


def residual_length(S_alpha, n_occs_tot, dS):
    """
    compute L(o) = L(t) + L(a) for all (a, t) in S_alpha
    i.e the length from a block of residual events

    Parameters
    ----------
    S_a: np.ndarray of shape or scalar
        array containing indices for the event `alpha` to consider

    n_occs_tot: int
        total number of occurrences in the original data

    dS: int
        max - min from original events
    """
    card = (
        S_alpha.shape[0] if isinstance(S_alpha, np.ndarray) else 1
    )  # TODO : remove me
    return log(dS + 1) - log(card / float(n_occs_tot))


def cycle_length(S, inter, n_event_tot, dS):
    """
    Parameters
    ----------
    S : np.array of type int64
        a collection of cycles, all having the same length : r
        The width of S is then r

    inter: np.array of type int64
        a collection of inter occurrence gaps, all having the same length: r - 1

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
    assert inter.shape[1] == r - 1  # check inter occurrences compliant with events
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
        a Series of occurrences
    n_tot: int
        total number of occurrences in the original events
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
        a Series of occurrences
    n_tot: int
        total number of occurrences in the original events
    max_length: int, default=None
        maximum number of occurrences for a cycle to cover,
        by default it will be set to :math:`\log_{2}\left(|S|\right)`

    """
    diffs = np.diff(S)
    triples = sliding_window_view(S, 3)
    diff_pairs = sliding_window_view(diffs, 2)
    dS = S.max() - S.min()

    score_one = residual_length(1, n_tot, dS)  # 1 really ?

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


def extract_triples(S, l_max=None):
    """
    Extract cycles of length 3 given a list of occurrences S
    Parameters
    ----------
    S: pd.Index
        input occurrences

    l_max: float
        maximum absolute difference between two occurrences to considered
        for inclusion in the same triple.
        By default it will be set to the median of the
        inter-occurrence differences from S.
    """
    triples = list()
    l_max = l_max or np.median(np.diff(S))

    # TODO : precompute diffs instead of for loop inner computation

    for idx, occ in enumerate(S[1:-1], 1):
        righties = S[idx + 1 :]
        lefties = S[:idx]
        righties_diffs = righties - occ
        lefties_diffs = lefties - occ
        dists = np.array(np.meshgrid(lefties_diffs, righties_diffs)).T.reshape(-1, 2)
        grid = np.abs(dists)
        keep = np.abs(grid[:, 0] - grid[:, 1]) < l_max
        t = occ + dists[keep]
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
    Reconstruct occurrences,
    starting from `start`, and
    correcting `period` with a delta for all deltas in `dE`,
    `len(dE)` occurrences are reconstructed

    Parameters
    ----------
    start: int or datetime
        starting point for the event
    period: int or timedelta
        period between two occurrences
    d_E: np.array of [int|timedelta]
        inters occurrences deltas
    """
    occurrences = [start]
    current = start
    for d_e in dE:
        e = current + period + d_e
        occurrences.append(e)
        current = e
    return occurrences


def _generate_candidates_batch(
    S_a: np.array, n_occs_tot: int, max_length=100, presort=False
):
    """
    Generate occurrences to be merged together for form a cycle
    The result if a list of np.array, sorted by decreasing order of width
    """
    if len(S_a) < 3:
        return list()
    if presort:
        S_a = np.sort(S_a)
    dS = S_a[-1] - S_a[0]
    l_max = np.log2(dS + 1) - 2
    cycles, covered = compute_cycles_dyn(S_a, n_occs_tot, max_length)
    covered = Bitmap(covered)

    if len(S_a) - len(covered) > 3:  # add triples if necessary
        _all = Bitmap(range(len(S_a)))
        _S_a = S_a[_all - covered]
        triples = extract_triples(_S_a, l_max)
        merged = merge_triples(triples)
        cycles.extend(merged)

    return list(sorted(cycles, key=lambda _: _.shape[1], reverse=True))


def generate_candidates(S_a, n_occs_tot, max_length=100):
    """
    Returns
    -------
    A list of candidate batches. Each batch contains candidates occurrences of the same length,
    hence stored in a common `numpy.ndarray`.
    Batches are sorted by decreasing order of width, so that we consider larger candidates first.
    """
    occ_batches = _generate_candidates_batch(S_a, n_occs_tot, max_length=max_length)
    res = list()
    for cand_batch in occ_batches:
        length = cand_batch.shape[1]
        E = np.diff(cand_batch, axis=1)
        period = np.floor(np.median(E, axis=1)).astype("int64")
        dE = (E.T - period).T
        tids = [
            Bitmap(_)
            for _ in np.searchsorted(S_a, cand_batch.reshape(-1)).reshape(
                cand_batch.shape
            )
        ]

        mdl_cost = sum(cycle_length(cand_batch, E, n_occs_tot, S_a[-1] - S_a[0]))
        df = pd.DataFrame(
            dict(
                start=cand_batch[:, 0],
                length=length,
                period=period,
                dE=dE.tolist(),
                tids=tids,
                cost=mdl_cost,
            )
        )
        res.append(df)

    res = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    return res


def evaluate(cands: pd.DataFrame, k: int):
    """
    Evaluate candidates `cands`, given S

    Unlike the original implementation by Galbrun & Al.,
    if an occurrence is present in more than one candidate cycle, we keep the cycle
    with the greatest length.

    Parameters
    ----------
    cands: pd.DataFrame
        a dataframe containing the candidates, as well-formed cycles
    k: int
        number of cycles to keep for the same occurence covered
    """
    idx = cands.explode("tids").groupby("tids").cost.nsmallest(k).index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(1)
    idx = idx.unique()
    return cands.loc[idx]


class SingleEventCycleMiner(BaseMiner):
    """
    Miner periodic cycles, but in the scope of a single event

    To this end, we only accept 1 dimensional arrays as input
    PeriodicCycleMiner will operate by instantiating one SingleEventCycleMiner per event.

    Parameters
    ----------
    max_length: int, default=20
        maximum length for a cycle
    keep_residuals: bool, default=False
        either to keep residuals (lossless compression) or not (lossy compression)
    n_occs_tot: int, default=None
        Number of total occurrences. When instanciating this class on a subset of the original
        data, this can be useful to pass the number of occurrences in this original data.
    k: int, default=2
        Maximum number of cycles covering the same occurrence
    """

    def __init__(self, max_length=20, keep_residuals=False, n_occs_tot=None, k=2):
        self.cycles_ = pd.DataFrame(columns=["start", "length", "period", "dE"])
        self.residuals_ = list()
        self.max_length = max_length
        self.keep_residuals = keep_residuals
        if n_occs_tot:
            assert isinstance(n_occs_tot, int)
        self.n_occs_tot = n_occs_tot
        self._dS = None  # difference between the last occurrence and the first
        self.k = k
        self.tid_pad = 0
        # TODO : set self.n_zeros_ at the scale of an event-projected miner like this one ?

    def fit(self, X):
        """
        Parameters
        ----------
        X: np.array of shape (n_occs, )
            Input occurrences

        Notes
        -----
        X will be reshaped to a 1-d vector in any case
        """
        X = np.array(X)  # COPY
        X = np.sort(X.reshape(-1))
        n_occs_tot = self.n_occs_tot or len(X)
        candidates = generate_candidates(X, n_occs_tot, max_length=self.max_length)

        if candidates.empty:
            warnings.warn(
                """could not generate candidates for the input sequence,
                the model is left empty"""
            )
            if self.keep_residuals:
                self.residuals_ = X
            return self

        cycles = evaluate(candidates, self.k)
        self.cycles_ = cycles
        if self.keep_residuals:
            # careful with `~`
            uncovered = Bitmap(range(len(X))) - Bitmap.union(*self.cycles_.tids)
            self.residuals_ = X[uncovered]

        # cycles.dE = cycles.dE.map(np.array) ?
        return self

    def discover(self):
        """
        Expose cycles as a brand new dataframe
        """
        return self.cycles_.copy(deep=True)

    evaluate = evaluate

    def reconstruct(self):
        """Reconstruct the original occurrences from the current cycles.
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
        alpha_occurrences = list()
        for start, period, dE in self.cycles_[["start", "period", "dE"]].values:
            occurrences = _reconstruct(start, period, dE)
            alpha_occurrences.extend(occurrences)
        alpha_occurrences.extend(self.residuals_)
        return np.array(alpha_occurrences)


class PeriodicCycleMiner(BaseMiner, MDLOptimizer):
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
    - :math:`p` is the inter-occurrence distance, called the `cycle period`
    - :math:`\\tau` is the index of the first occurrence, called the `cycle starting point`
    - :math:`E` is a list of :math:`r - 1` signed integer offsets, i.e `cycle shift corrections`


    Parameters
    ----------
    max_length: int, default=20
        maximum length for a candidate cycle, when running the dynamic programming heuristic
    keep_residuals: bool, default=False
        Either to keep track of residuals (occurrences not covered by any cycle) or not.
        Residuals are required for a lossless reconstruction of the original data.
    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single event is attributed a job
        to discover potential cycles.
        Threads are preffered over processes.

    Examples
    --------
    >>> from skmine.periodic import PeriodicCycleMiner
    >>> import pandas as pd
    >>> S = pd.Series("ring_a_bell", [10, 20, 32, 40, 60, 79, 100, 240])
    >>> pcm = PeriodicCycleMiner().fit(S)
    >>> pcm.discover()
                   start  length  period       cost
    ring_a_bell 1     10       3      11  23.552849
                0     40       4      20  24.665780

    References
    ----------
    .. [1]
        Galbrun, E & Cellier, P & Tatti, N & Termier, A & Crémilleux, B
        "Mining Periodic Pattern with a MDL Criterion"
    """

    def __init__(self, *, max_length=20, keep_residuals=False, n_jobs=1):
        self.miners_ = dict()
        self.is_datetime_ = None
        self.n_zeros_ = 0
        self.n_jobs = n_jobs
        self.max_length = max_length
        self.keep_residuals = keep_residuals

    def fit(self, S):
        """fit PeriodicCycleMiner on data logs

        This generate new candidate cycles and evaluate them.
        Residual occurrences are stored as an internal attribute,
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
            warnings.warn("found duplicates in input sequence, removing them")
            S = S.groupby(S.index).first()

        S = S.copy()
        S.index, self.n_zeros_ = _remove_zeros(S.index.astype("int64"))
        # TODO : do this in SingleEventCycleMiner?

        n_occs_tot = S.shape[0]
        alpha_groups = S.groupby(S.values).groups
        miners = {
            k: SingleEventCycleMiner(self.max_length, self.keep_residuals, n_occs_tot)
            for k in alpha_groups.keys()
        }
        self.miners_ = {k: miners[k].fit(v) for k, v in alpha_groups.items()}

        for (event, miner,) in self.miners_.items():
            if "tids" in miner.cycles_.columns:
                # FIXME: this is highly inefficient
                miner.cycles_.tids = miner.cycles_.tids.map(
                    lambda tids: Bitmap(
                        np.searchsorted(S.index, alpha_groups[event][tids])
                    )
                )

        return self

    evaluate = SingleEventCycleMiner.evaluate

    def discover(self, shifts=False, tids=False):
        """Return cycles as a pandas DataFrame, with 3 columns,
        with a 2-level multi-index: the first level mapping events,
        and the second level being positional

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  ======================================
                start       when the cycle starts
                length      number of occurrences in the event
                period      inter-occurrence delay
                dE          shift corrections, if shifts=True
                tids        Transactions ids covered, if tids=True
                cost        MDL cost
                ==========  ======================================

        Example
        -------
        >>> from skmine.periodic import PeriodicCycleMiner
        >>> import pandas as pd
        >>> S = pd.Series("ring", [10, 20, 32, 40, 60, 79, 100, 240])
        >>> pcm = PeriodicCycleMiner().fit(S)
        >>> pcm.discover()
                start  length  period       cost
        ring 1     10       3      11  23.552849
             0     40       4      20  24.665780
        """
        all_cols = ["start", "length", "period", "cost"]
        if tids:
            all_cols.extend(["tids"])
        if shifts:
            all_cols.extend(["dE"])

        series = list()
        for miner in self.miners_.values():
            disc = miner.discover()
            if not disc.empty:
                series.append(disc)

        if not series:
            return pd.DataFrame()  # FIXME
        cycles = pd.concat(series, keys=self.miners_.keys())[all_cols]
        cycles.loc[:, ["start", "period"]] = cycles[["start", "period"]] * (
            10 ** self.n_zeros_
        )

        if shifts:
            cycles.loc[:, "dE"] = cycles.dE.map(np.array) * (10 ** self.n_zeros_)

        if self.is_datetime_:
            cycles.loc[:, "start"] = cycles.start.astype("datetime64[ns]")
            cycles.loc[:, "period"] = cycles.period.astype("timedelta64[ns]")

        return cycles

    def reconstruct(self):
        """Reconstruct the original occurrences from the current cycles.
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
        series = [
            pd.Series(event, index=miner.reconstruct())
            for event, miner in self.miners_.items()
        ]
        S = pd.concat(series)
        S.index *= 10 ** self.n_zeros_
        if self.is_datetime_:
            S.index = S.index.astype("datetime64[ns]")

        # TODO : this is due to duplicate event in the cycles, handle this in .evaluate()
        S = S.groupby(S.index).first()

        return S

    def generate_candidates(self, S):
        """
        Parameters
        ----------
        S: pd.Index or numpy.ndarray
            Series of occurrences for a specific event

        Returns
        -------
        dict[object, list[np.ndarray]]
            A dict, where each key is an event and each value a list of batch of candidates.
            Batches are sorted in inverse order of width,
            so that we consider larger candidate cycles first.
        """
        # TODO only for InteractiveMode

    def get_residuals(self):
        """Get the residual events, i.e events not covered by any cycle

        It is the complementary function to `discover`

        Returns
        -------
        pd.Series
            residual events
        """
        series = [
            pd.Series(event, index=miner.residuals_)
            for event, miner in self.miners_.items()
        ]
        residuals = pd.concat(series)
        if not residuals.empty:
            residuals.index *= 10 ** self.n_zeros_
        if self.is_datetime_:
            residuals.index = residuals.index.astype("datetime64[ns]")
        return residuals
