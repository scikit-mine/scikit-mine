"""
The :mod:`skmine.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""
from functools import partial
from ._samples_generator import make_transactions

from ._base import get_data_home
_REQ_TIMEOUT = 8


try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

urlopen = partial(urlopen, timeout=_REQ_TIMEOUT)
