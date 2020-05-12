"""
IO configuration for skmine.datasets methods
"""
from functools import partial

_REQ_TIMEOUT = 8

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

urlopen = partial(urlopen, timeout=_REQ_TIMEOUT)
