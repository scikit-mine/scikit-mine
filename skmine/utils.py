"""
utils functions
"""

import numbers
from itertools import count

import numpy as np
import pandas as pd
import scipy.sparse
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch
from numpy.lib.stride_tricks import as_strided
from sortedcontainers import SortedList


def _check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a numpy.random.RandomState instance"
    )


def _check_min_supp(min_supp, accept_absolute=True):
    if isinstance(min_supp, int):
        if not accept_absolute:
            raise ValueError(
                "Absolute support is prohibited, please provide a float value between 0 and 1"
            )
        if min_supp < 1:
            raise ValueError("Minimum support must be strictly positive")
    elif isinstance(min_supp, float):
        if min_supp < 0 or min_supp > 1:
            raise ValueError("Minimum support must be between 0 and 1")
    else:
        raise TypeError("Mimimum support must be of type int or float")
    return min_supp


def _check_growth_rate(gr):
    if not gr > 1:
        raise ValueError("growth ratio should be greater than 1")
    return gr


def filter_maximal(itemsets):
    """filter maximal itemsets from a set of itemsets

    Parameters
    ----------
    itemsets: Iterator[list]
        a list of itemsets

    Returns
    -------
    SortedList
    """
    maximals = SortedList(key=len)
    itemsets = sorted(itemsets, key=len, reverse=True)
    for itemset_set in itemsets:
        gts = maximals.irange(itemset_set)
        # # is there a superset amongst bigger itemsets ?
        if not any(map(lambda e: e > itemset_set, gts)):
            maximals.add(itemset_set)  # O(log(len(maximals)))

    return maximals


def filter_minimal(itemsets):
    """filter minimal itemsets from a set of itemsets

    Parameters
    ----------
    itemsets: Iterator[frozenset]
        a set of itemsets

    Returns
    -------
    SortedList
    """
    minimals = SortedList(key=len)
    itemsets = sorted(itemsets, key=len)
    for iset in itemsets:
        lts = minimals.irange(None, iset)
        # is there a subset amongst the smaller itemsets ?
        if not any(map(lambda e: e < iset, lts)):
            minimals.add(iset)

    return minimals


def supervised_to_unsupervised(D, y):
    """
    for sklearn compatibility, eg. sklearn.multiclass.OneVSRest

    Parameters
    ----------
    D: pd.DataFrame
        input transactional dataset

    y: np.ndarray of shape (n_samples,)
        corresponding labels
    """
    mask = np.where(y.reshape(-1))[0]
    D = D.iloc[mask]

    return D


def _check_D_sklearn(D):
    if object in D.dtypes.values:  # SKLEARN : check_dtype_object
        raise TypeError("argument must be a string or a number")

    if D.shape[1] == 0:  # SKLEARN : check_empty_data_messages
        raise ValueError("Empty data")

    pd.options.mode.use_inf_as_na = True
    if D.isnull().values.any():
        raise ValueError("esimator does not check for NaN and inf")
    pd.options.mode.use_inf_as_na = False


def _check_D(D):
    if isinstance(D, pd.DataFrame):
        D = D.reset_index(drop=True)  # positional indexing
    elif isinstance(D, np.ndarray):
        D = pd.DataFrame(D)
    elif scipy.sparse.issparse(D):
        D = pd.DataFrame.sparse.from_spmatrix(D)
    else:
        raise TypeError("D should be an instance of np.ndarray or pd.DataFrame")

    _check_D_sklearn(D)

    return D


def _check_y(y):
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y should be an instance of np.ndarray or pd.Series")

    # TODO : pd.Categorical
    return y


def _check_D_y(D, y=None):
    D = _check_D(D)
    if y is not None:
        y = _check_y(y)
    return D, y


def intersect2d(ar1, ar2, return_indices=True):
    """
    Find the intersection of two 2 dimnesional arrays

    Return the sorted, unique rows that are both of the input arrays

    Parameters
    ----------
    x, y: array_like
        Input arrays
    return_indices: bool
        If True, the indices wich correspond to the intersection of the two array
        are returned. The first instance of a value is used if there are multiple.
        Default is False

    Returns
    -------
    intersect2d: ndarray
        Sorted 2D array of common and unique rows
    comm1: ndarray
        The indices of the first occurences of the common rows in `ar1`.
        Only provided if `return_indices` is True
    comm2: ndarray
        The indices of the first occurences of the common rows in `ar2`.
        Only provided if `return_indices` is True
    """
    ar1, ar2 = np.asanyarray(ar1), np.asanyarray(ar2)
    assert ar1.ndim == ar2.ndim == 2
    _, x_ind, y_ind = np.intersect1d(ar1[:, 0], ar2[:, 0], return_indices=True)
    x, y = ar1[x_ind], ar2[y_ind]
    xy_where = np.argwhere((x == y).all(axis=1)).reshape(-1)
    x_ind = x_ind[xy_where]
    if return_indices:
        y_ind = y_ind[xy_where]
        return ar1[x_ind], x_ind, y_ind
    return ar1[x_ind]


def _sliding_window_view_dispatcher(
    x, window_shape, axis=None, *, subok=None, writeable=None
):
    return (x,)


@array_function_dispatch(_sliding_window_view_dispatcher)
def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """
    COPIED from https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/stride_tricks.py#L122-L336

    Create a sliding window view into the array with the given window shape.
    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.

    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.
    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.
    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:
    - `scipy.signal.fftconvolve`
    - filtering functions in `scipy.ndimage`
    - moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.
    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.
    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.
    Examples
    --------

    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])
    """
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def bron_kerbosch(candidates: dict, clique=None, excluded=None, depth=0):
    """
    Bron-Kerbosch algorithm, from https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm

    Parameters
    ----------
    candidates: dict[object, list[object]]
        a mapping from each node to its existing neighbours
    """
    if not candidates and not excluded and clique is not None and len(clique) > 2:
        yield [_ for _ in clique]
        return

    # pass None as default arg instead of dict()
    # fix collisions in pytest, really obscure
    clique = clique or dict()
    excluded = excluded or dict()

    if depth > 20:
        return

    for node, neighbours in list(candidates.items()):
        new_clique = {**{node: neighbours}, **clique}
        new_candidates = {k: v for k, v in candidates.items() if k in neighbours}
        new_excluded = {k: v for k, v in excluded.items() if k in neighbours}

        yield from bron_kerbosch(new_candidates, new_clique, new_excluded)

        del candidates[node]
        excluded[node] = neighbours

