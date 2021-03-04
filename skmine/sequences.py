import warnings
from collections import defaultdict
import pandas as pd
import numpy as np

from .bitmaps import Bitmap
from .base import BaseMiner, MDLOptimizer


def to_vertical(D):
    """
    translate a sequential dataset to its vertical representation
    """
    event_to_indices = defaultdict(Bitmap)
    seq_lengths = [0]
    idx = 0
    for s in D:
        seq_lengths.append(len(s))
        for e in s:
            event_to_indices[e].add(idx)
            idx += 1

    return dict(event_to_indices), np.cumsum(seq_lengths)


def compress_size(D, patt, seq_lengths, *, max_gap=1):
    """Algorithm 2 from original paper

    This version uses vertical representations of the datasets
    "Pointers" mentioned in the paper are bitmaps pointing to start positions

    Rough complexity is O(|supp(patt[0])| + |patt|)

    Parameters
    ----------
    D: dict[object, Bitmap]
        vertical representation of D
    patt: list
        Pattern to evaluate
    seq_lengths: np.array of shape(n_transactions, )
        a 1-D array containing transaction lengths

    max_gap: int
        maximum number of gaps in a pattern

    TODO add max_gap=2 and interleaving=True
    """
    cost = len(patt) + sum(map(len, D.values()))
    # sum(D.values()) vertical eq. to sum(len(Si) for Si in horizontal D)
    D = {
        k: D[k].copy() for k in patt
    }  # check in pattern, other events are left untouched anyway
    patt = tuple(patt)

    start_positions = D[patt[0]].copy()  # iter over positions for first element of p
    start_seq_indices = np.searchsorted(seq_lengths, start_positions, side="right")
    seq_end_indices = seq_lengths[start_seq_indices]
    assert (seq_end_indices >= start_seq_indices).all()
    for start_pos, seq_end_idx in zip(start_positions, seq_end_indices):
        pos = start_pos
        indices = [pos]
        gap_ctr = max_gap
        for e in patt[1:]:
            bits = D[e].clamp(pos + 1, min(pos + gap_ctr + 1, seq_end_idx))
            if bits:
                next_set_bit = bits.min()
                gap_ctr -= next_set_bit - pos - 1  # consume gap
                indices.append(next_set_bit)
                pos = next_set_bit
        if len(indices) == len(patt):  # found entire pattern
            bm = Bitmap(indices)  # TODO use regular lists
            for e in patt:
                D[e] -= bm
            if patt not in D:
                D[patt] = Bitmap([bm.min()])
            else:
                D[patt].add(bm.min())  # add pointer to start of pattern

    # cost updating can be done at the end, thx to vertical representation
    if patt in D:
        cost -= len(D[patt]) * (len(patt) - 1)

    return cost, D


def best_compressing(D, seq_lengths, *, max_gap=1):
    """
    get the best compressing pattern from the dataset D
    """
    p = list()
    b = np.inf
    update_d = dict()

    while len(D) - len(p) > 0:
        cands = sorted(D.keys() - p)  # lexicographical order
        e = max(
            cands, key=lambda k: (len(D[k]), estimate_gain(D[k], seq_lengths))
        )  # TODO : lazy estimate gain
        p.append(e)
        comp_size, u_d = compress_size(D, p, seq_lengths, max_gap=max_gap)
        if comp_size < b:
            b = comp_size
            update_d = u_d
        else:
            if len(p) > 1:
                p.pop(-1)
            break
    return tuple(p), update_d  # TODO : return b ?


def estimate_gain(tids, seq_lengths):
    """
    Let e be the event associated with tids
    the biggest the dists between each occurence of e, the better

    We take sequences lengths into account, so we dont mistakenly favor
    events from different transactions
    """
    diffs = []
    for i in range(len(seq_lengths) - 1):
        start, end = seq_lengths[i : i + 2]
        _tids = tids.clamp(start, end)
        diff = np.diff(_tids)
        diffs.append(diff.mean())
    return np.mean(diffs)


class GoKrimp(BaseMiner, MDLOptimizer):  # TODO : inerit MDL Optimizer
    def __init__(self, k=100):
        self.k_ = k
        self._cum_seq_lengths = None
        self.codetable_ = None
        self.standard_codetable_ = None

    def fit(self, D):
        vert_D, cum_seq_lengths = to_vertical(D)
        self.standard_codetable_ = {
            event: tids.copy() for event, tids in vert_D.items()
        }

        H = dict()
        while len(H) < self.k_:
            p_star, update_d = best_compressing(vert_D, cum_seq_lengths)
            if not p_star:
                break
            H[p_star] = update_d.pop(p_star)
            vert_D.update(update_d)
            empties = [e for e, tids in vert_D.items() if not tids]
            for e in empties:
                del vert_D[e]
        self.codetable_ = H
        self._cum_seq_lengths = cum_seq_lengths

        return self

    def discover(self):
        cpy = {pattern: tids.copy() for pattern, tids in self.codetable_.items()}
        return pd.Series(cpy.values(), index=list(cpy.keys()))

    def generate_candidates(self):
        """
        Finds the most compressing pattern given the current state
        of the vertical representation of the data.

        Notes
        -----
        Algorithm 4 from original paper, namely BestCompressing
        see section 5.2.2 "The most compressing pattern"
        """
        yield best_compressing(self.codetable_, self.seq_lengths)

    def evaluate(self, patt):
        pass  # TODO

    def cover(self, D):
        pass  # TODO

    def reconstruct(self):
        """
        1. reconstruct one big sequence
        2. split it into the original sequences giving the cumulative sums of the transaction lengths
        """
        warnings.warn(
            """
            GoKRIMP has a lossy compression scheme,
            hence reconconstrution is performed from the
            standard codetable
        """
        )
        # last of cumulative sequence lengths if the total size of the flattened sequence
        flat = np.empty(self._cum_seq_lengths[-1], dtype=object)
        flat[:] = None
        for event, tids in self.standard_codetable_.items():
            flat[tids] = event

        l = list()
        for i in range(len(self._cum_seq_lengths) - 1):
            start, end = self._cum_seq_lengths[i : i + 2]
            l.append(flat[start:end].tolist())
        return pd.Series(l)
