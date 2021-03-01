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

from skmine.base import MDLOptimizer, BaseMiner
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

        second_half_ent = get_entropy_nb_ones(y[idx:end])[0]
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

    generate_candidates = generate_cut_point

    def evaluate(self, y, start, end, cut_point):
        """
        Evaluate vector y of size ``end`` - ``start``,
        given a ``cutpoint``
        """
        entropy1, k1 = get_entropy_nb_ones(y[start:cut_point])
        entropy2, k2 = get_entropy_nb_ones(y[cut_point:end])
        whole_entropy, k0 = get_entropy_nb_ones(y[start:end])

        N = end - start

        part1 = 1 / N * ((cut_point - start) * entropy1 + (end - cut_point) * entropy2)
        delta = np.log2(pow(3, k0) - 2) - (
            k0 * whole_entropy - k1 * entropy1 - k2 * entropy2
        )

        gain = whole_entropy - part1

        return gain > 1 / N * (np.log2(N - 1) + delta)

    def fit(self, X, y):
        """
        fit discretizer on a single feature

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples,)
            Input vector to fit the discretizer on

        y : numpy.ndarray of shape (n_samples,)
            Labels to fit the discretizer on
        """
        assert len(X.shape) == 1
        assert np.issubdtype(y.dtype, np.integer)

        order = np.argsort(X)
        X = X[order]
        y = y[order]

        cut_points = set()
        search_intervals = [(0, len(X), 0)]
        while search_intervals:
            level = search_intervals.pop(-1)  # pop back
            start, end, depth = level

            k = generate_cut_point(y, start, end)

            is_better = self.evaluate(y, start, end, k)
            if k == -1 or not is_better:
                front = -np.inf if start == 0 else (X[start - 1] + X[start]) / 2
                back = np.inf if end == len(X) else (X[end - 1] + X[end]) / 2

                if front != -np.inf:
                    cut_points.add(front)
                if back != np.inf:
                    cut_points.add(back)
            else:
                search_intervals.append((start, k, depth + 1))
                search_intervals.append((k, end, depth + 1))

        self.cut_points_ = np.array(list(cut_points))
        return self


class MDLPDiscretizer(BaseMiner):
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

    Attributes
    ----------
    cut_points_: dict
        A mapping between columns and their respective cut points.
        If fitted on a pandas DataFrame, keys will be the DataFrame column names.


    References
    ----------
    Usama M. Fayyad, Keki B. Irani
    "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning",
    1993

    Examples
    --------

    >>> from skmine.preprocessing import MDLPDiscretizer
    >>> from sklearn.datasets import load_iris  # doctest: +SKIP
    >>> iris = load_iris()                      # doctest: +SKIP
    >>> X, y = iris.data, iris.target           # doctest: +SKIP
    >>> disc = MDLPDiscretizer()                # doctest: +SKIP
    >>> disc.fit(X, y)                          # doctest: +SKIP
    >>> disc.cut_points_                        # doctest: +SKIP
    {0: array([5.5, 6.2]), 1: array([2.9, 3.3]), 2: array([2.45, 4.9 ]), 3: array([0.8, 1.7])}

    """

    def __init__(self, random_state=None, n_jobs=1):
        self.cut_points_ = dict()
        self.random_state = _check_random_state(random_state)
        self.n_jobs = n_jobs
        self.discretizers_ = []

    def fit(self, X, y):
        """fit the MLDP discretizer on an input matrix ``X``, given a label vector ``y``.

        Parameters
        ----------
        X: np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            The input matrix containing features. A set of cut points
            will be affected to each feature

        y : np.ndarray of pd.Series of shape(n_samples,)
            The label vector used to discretize ``X``
        """
        assert y is not None and np.issubdtype(y.dtype, np.integer)
        permutation = self.random_state.permutation(len(X))
        _X = X.values if isinstance(X, pd.DataFrame) else X
        _y = y.values if isinstance(y, pd.Series) else y
        _X = _X[permutation]
        _y = _y[permutation]

        n_cols = _X.shape[1]

        discs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(MDLPVectDiscretizer().fit)(_X[:, idx], _y) for idx in range(n_cols)
        )

        self.discretizers_ = discs
        cut_points = [d.cut_points_ for d in discs]

        if isinstance(X, pd.DataFrame):
            self.cut_points_ = dict(zip(X.columns, cut_points))
        else:
            self.cut_points_ = dict(enumerate(cut_points))

        return self

    @property
    def codetable(self):  # FIXME : this should be inherited from MDL
        """user-friendly view on cut points"""
        return pd.Series(self.cut_points_)

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        """Discretizes the input matrix X

        This applies the cutpoints their respective columns
        """
        if isinstance(X, pd.DataFrame) and not set(self.cut_points_) == set(X.columns):
            raise ValueError(f"X columns should be {self.cut_points_.keys()}")
        _X = X.values if isinstance(X, pd.DataFrame) else X

        vects = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(np.searchsorted)(cut_points, _X[:, idx])
            for idx, cut_points in enumerate(self.cut_points_.values())
        )

        vects = [v.reshape(len(v), 1) for v in vects]
        _X = np.concatenate(vects, axis=1)

        if isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(_X, columns=X.columns)

        return _X

    def fit_transform(self, X, y=None):
        "fit on X and y, then transform X"
        return self.fit(X, y).transform(X)
