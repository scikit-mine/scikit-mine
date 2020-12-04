"""
Base IO for all periodic datasets
"""
import os

import pandas as pd

from ._base import get_data_home


def fetch_health_app(data_home=None, filename="health_app.csv"):
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, index_col=0, squeeze=True, dtype="string")
    if filename in os.listdir(data_home):
        s = pd.read_csv(p, index_col=0, squeeze=True)
    else:
        s = pd.read_csv(
            "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log",
            sep="|",
            error_bad_lines=False,
            usecols=[0, 1],
            **kwargs
        )
        s.to_csv(p)
    s.index.name = "timestamp"
    s.index = pd.to_datetime(s.index, format="%Y%m%d-%H:%M:%S:%f")

    return s
