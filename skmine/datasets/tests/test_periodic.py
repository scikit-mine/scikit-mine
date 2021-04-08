import os

import pandas as pd
import pytest

from ..periodic import fetch_canadian_tv, fetch_health_app


def mock_read_csv(*args, **kwargs):
    return pd.Series([2, 3], index=["20171223-22:15:29:606", "20171223-22:15:29:615"])


def mock_read_csv_canadian_tv(*args, **kwargs):
    programs = ["The Moblees", "Big Block Sing Song", "Big Block Sing Song", "CBC Kids"]
    index = pd.date_range(start="08/01/2020", periods=len(programs), freq="1H")
    return pd.Series(programs, index=index)


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
    d_file = ["canadian_tv.txt"] if already_downloaded else []
    if not already_downloaded:
        monkeypatch.setattr(pd.Series, "to_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(os, "listdir", lambda *args: d_file)
    monkeypatch.setattr(pd, "read_csv", mock_read_csv_canadian_tv)
    data = fetch_canadian_tv()
    assert data.shape == (4,)
    assert isinstance(data.index, pd.DatetimeIndex)
