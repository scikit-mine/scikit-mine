"""
Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning
"""

# Author: Rémi Adon <remi.adon@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy

from skmine.base import MDLOptimizer
from skmine.utils import _check_random_state


def get_entropy_nb_ones(y):
    """
    for a given vector y, returns the entropy (base 2) and the
    number of non zeros values in the one-hot-encoded counting vector
    associated with y
    """
    ohe = np.bincount(y) / len(y)  # counts, one hot encoded
    return entropy(ohe, base=2), np.sum(ohe != 0)


def generate_cut_point(y, start, end):
    """
    Generate a cut point given a label vector ``y``, a start position
    ``start`` and a final position ``end``

    Starts with an infinite entropy, then iteratively moves a index inside
    ``y``, pre-evaluates entropy on both sides of this index.

    It returns the index with the minimum entropy.
    """
    length = end - start
    ent = np.inf
    k = -1

    for idx in range(start + 1, end):
        if y[idx - 1] == y[idx]:
            continue

        first_half_ent = get_entropy_nb_ones(y[start:idx])[0]
        first_half_ent *= (idx - start) / length

        second_half_ent = get_entropy_nb_ones(y[idx: end])[0]
        second_half_ent *= (end - idx) / length

        new_ent = first_half_ent + second_half_ent

        if new_ent < ent:
            ent = new_ent
            k = idx

    return k


class MDLPVectDiscretizer(MDLOptimizer):
    """
    Basic block for the implementation of
    "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning".

    This class operates at a column level, i.e it finds the best cut points for a given feature
    to fit a the corresponding labels
    """
    def __init__(self, min_depth=0):
        self.min_depth_ = min_depth
        self.entropy_ = np.inf
        self.cut_points_ = np.array([])

    def evaluate_gain(self, y, start, end, cut_point):
        entropy1, k1 = get_entropy_nb_ones(y[start: cut_point])
        entropy2, k2 = get_entropy_nb_ones(y[cut_point: end])
        whole_entropy, k0 = get_entropy_nb_ones(y[start: end])

        N = end - start

        part1 = 1 / N * ((cut_point - start) * entropy1 + (end - cut_point) * entropy2)
        gain = whole_entropy - part1
        entropy_diff = k0 * whole_entropy - k1 * entropy1 - k2 * entropy2
        delta = np.log2(pow(3, k0) - 2) - entropy_diff

        return gain > 1 / N * (np.log2(N - 1) + delta)

    def fit(self, X, y):
        assert len(X.shape) == 1

        order = np.argsort(X)
        X = X[order]
        y = y[order]

        cut_points = set()
        search_intervals = [(0, len(X), 0)]
        while search_intervals:
            level = search_intervals.pop(len(search_intervals) - 1)  # pop back
            start, end, depth = level

            k = generate_cut_point(y, start, end)

            is_better = self.evaluate_gain(y, start, end, k)
            if k == -1 or not is_better:
                front = -np.inf if start == 0 else (X[start - 1] + X[start]) / 2
                back = np.inf if end == len(X) else (X[end - 1] + X[end]) / 2

                if front == back: continue
                if front != -np.inf:
                    cut_points.add(front)
                if back != np.inf:
                    cut_points.add(back)
            else:
                search_intervals.append((start, k, depth + 1))
                search_intervals.append((k, end, depth + 1))


        self.cut_points_ = np.array(list(cut_points))
        return self


class MDLPDiscretizer():
    """
    Implementation of "Multi-Interval Discretization of Continuous-Valued Attributes
    for Classification Learning".

    Given class labels ``y``, MDLPDIscretizer discretizes continuous variables from
    ``X`` by minimizing the entropy in each interval.

    Parameters
    ----------
    random_state : int, RandomState instance, default=None
        random state to use to shuffle the data. Can affect the outcome, leading to
        slightly different cut points if a variable contains samples with the same value
        but different labels.

    References
    ----------
    .. [1]
        Usama M. Fayyad, Keki B. Irani
        "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning",
        1993

    Attributes
    ----------
    cut_points_: dict
        A mapping between columns and their respective cut points.
        If fitted on a pandas DataFrame, keys will be the DataFrame column names.

    Examples
    --------
    # TODO
    """
    def __init__(self, random_state=None, n_jobs=1):
        self.cut_points_ = dict()
        self.random_state = _check_random_state(random_state)
        self.n_jobs = n_jobs
        self.discretizers_ = []

    def __repr__(self):
        return repr(self.cut_points_)

    def fit(self, X, y):
        permutation = self.random_state.permutation(len(y))
        X = X[permutation]
        y = y[permutation]

        vals = X.values if isinstance(X, pd.DataFrame) else X
        n_cols = vals.shape[1]

        discs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(MDLPVectDiscretizer().fit)(vals[:, idx], y) for idx in range(n_cols)
        )

        self.discretizers_ = discs
        cut_points = [d.cut_points_ for d in discs]

        if isinstance(X, pd.DataFrame):
            self.cut_points_ = dict(zip(X.columns, cut_points))
        else:
            self.cut_points_ = dict(enumerate(cut_points))

        return self
