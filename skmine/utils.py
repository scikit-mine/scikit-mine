"""
utils functions
"""

import numpy as np
from collections import defaultdict

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
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        raise TypeError('random_state should be an int or a RandomState instance')

    return random_state
