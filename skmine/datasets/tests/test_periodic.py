from ..periodic import fetch_health_app
import os
import pandas as pd


def mock_read_csv(*args, **kwargs):
    return pd.Series([2, 3], index=["20171223-22:15:29:606", "20171223-22:15:29:615"])


def test_fetch_any(monkeypatch):
    name = "filename"
    monkeypatch.setattr(pd.Series, "to_csv", lambda *args, **kwargs: None)

    # force file to be in DATA_HOME, so we don't need to fetch it
    monkeypatch.setattr(os, "listdir", lambda *args: ["{}.csv".format(name)])
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    data = fetch_health_app(filename=name)
    assert data.shape == (2,)
    assert isinstance(data.index, pd.DatetimeIndex)

    # now DATA_HOME to be empty, file has to be fetched
    monkeypatch.setattr(os, "listdir", lambda *args: list())
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    data = fetch_health_app()
    assert data.shape == (2,)
    assert isinstance(data.index, pd.DatetimeIndex)
