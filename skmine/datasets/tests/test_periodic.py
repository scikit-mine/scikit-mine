import os

import pandas as pd
import pytest

from .. import periodic
from ..periodic import fetch_canadian_tv, fetch_health_app


def mock_read_csv(*args, **kwargs):
    return pd.Series([2, 3], index=["20171223-22:15:29:606", "20171223-22:15:29:615"])


def mock_read_csv_canadian_tv(*args, **kwargs):
    return pd.Series(
        [
            "2PGR10CBC   200801060000061130001130The Moblees                                       1-16 Great Galloping Moblees                                  C3772721 35105ACD",
            "2PGR10CBC   200801061130061330000200Big Block Sing Song                               2-009 CHICKENS                                                B3684421 351120CD",
            "2PGR10CBC   200801061330061530000200Big Block Sing Song                               2-010 SPECTACULAR                                             B3684421 351120CD",
            "2PGR10CBC   200801061530061535000005CBC Kids                                          19/20-3456, Coming Up - NapkinMan                 CBC               21 341120CC",
        ]
    )


@pytest.mark.parametrize("already_downloaded", [True, False])
def test_fetch_healt_app(monkeypatch, already_downloaded):
    name = "filename"
    monkeypatch.setattr(pd.Series, "to_csv", lambda *args, **kwargs: None)

    # force file to be in DATA_HOME, so we don't need to fetch it
    if already_downloaded:
        monkeypatch.setattr(os, "listdir", lambda *args: ["{}.csv".format(name)])
        monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    else:  # now DATA_HOME to be empty, file has to be fetched
        monkeypatch.setattr(os, "listdir", lambda *args: list())
        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    data = fetch_health_app(filename=name)
    assert data.shape == (2,)
    assert isinstance(data.index, pd.DatetimeIndex)


@pytest.mark.parametrize("already_downloaded", [True, False])
def test_fetch_canadian_tv(monkeypatch, already_downloaded):
    if already_downloaded:
        monkeypatch.setattr(
            os, "listdir", lambda *args: ["2020-08\CBC_202008_140114251.log"]
        )
    else:
        monkeypatch.setattr(os, "listdir", lambda *args: [])
        monkeypatch.setattr(periodic, "_extract_canadian_tv", lambda *args: None)
    monkeypatch.setattr(pd, "read_csv", mock_read_csv_canadian_tv)
    data = fetch_canadian_tv()
    assert data.shape == (4,)
    assert isinstance(data.index, pd.DatetimeIndex)
