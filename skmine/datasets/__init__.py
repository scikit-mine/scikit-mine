_REQ_TIMEOUT = 8
from functools import partial

from ._base import get_data_home

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

urlopen = partial(urlopen, timeout=_REQ_TIMEOUT)
