"""
The :mod:`skmine.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""
from functools import partial
from ._samples_generator import make_transactions
from ._samples_generator import make_classification
from ._instacart import fetch_instacart

from ._base import get_data_home
