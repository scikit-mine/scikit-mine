"""
utils functions
"""

from collections import defaultdict
from functools import partial
from operator import gt, lt

import numpy as np


class lazydict(defaultdict):
    """
    lazydict(default_factory[, ...]) --> dict with default factory

    The default factory is called with key as argument to produce
    a new value (via  __getitem__ only), and store it.
    A lazydict compares equal to a dict with the same items.
    All remaining arguments are treated the same as if they were
    passed to the dict constructor, including keyword arguments.
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        res = self[key] = self.default_factory(key)
        return res


def _check_random_state(random_state):
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.random.RandomState):
        raise TypeError('random_state should be an int or a RandomState instance')

    return random_state


def _check_min_supp(min_supp, accept_absolute=True):
    if isinstance(min_supp, int):
        if not accept_absolute:
            raise ValueError(
                'Absolute support is prohibited, please provide a float value between 0 and 1'
            )
        if min_supp < 1:
            raise ValueError('Minimum support must be strictly positive')
    elif isinstance(min_supp, float):
        if min_supp < 0 or min_supp > 1:
            raise ValueError('Minimum support must be between 0 and 1')
    else:
        raise TypeError('Mimimum support must be of type int or float')
    return min_supp


def _check_growth_rate(gr):
    if not gr > 1:
        raise ValueError('growth ratio should be greater than 1')
    return gr


def filter_within(itemsets, op):
    """
    Filter patterns within a patternset,
    comparing each pattern to every other patterns,
    by applying ``op``.

    Notes
    -----
        O(n²) complexity
    """
    itemsets = [set(e) for e in itemsets]
    for iset in itemsets:
        if any(map(lambda e: op(e, iset), itemsets)):
            continue
        yield iset


filter_maximal = partial(filter_within, op=gt)
filter_minimal = partial(filter_within, op=lt)
