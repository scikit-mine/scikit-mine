import os

import pytest

from skmine.datasets import get_data_home


def test_data_home():
    assert os.path.exists(get_data_home())
