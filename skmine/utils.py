"""
utils functions
"""

from collections import defaultdict

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


def _check_min_supp(min_supp):
    if isinstance(min_supp, int):
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
