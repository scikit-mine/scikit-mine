import os
from datetime import datetime
from unittest import mock
from unittest.mock import patch, mock_open

import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ..periodic import fetch_canadian_tv, fetch_health_app, fetch_file, read_ubiq_user


def mock_read_csv(*args, **kwargs):
    return pd.DataFrame([2, 3], index=["20171223-22:15:29:606", "20171223-22:15:29:615"])


def mock_read_csv_canadian_tv(*args, **kwargs):
    programs = ["The Moblees", "Big Block Sing Song", "Big Block Sing Song", "CBC Kids"]
    index = pd.date_range(start="08/01/2020", periods=len(programs), freq="1H")
    return pd.DataFrame(programs, index=index)


def mock_read_csv_ubiq(*args, **kwargs):
    events = ["com.android.launcher_I", "com.android.settings_I", "com.motorola.context_I",
              "com.motorola.modemservice_S"]
    index = pd.date_range(start="14/11/2013", periods=len(events), freq="1S")
    s = pd.Series(events, index=index)
    s.index.name = 'timestamp'
    s.name = 'USER_0'
    return pd.Series(events, index=index)


def test_fetch_file_two_columns():
    custom_dataset = "2020-08-01 06:00:00,The Moblees\n" \
                     "2020-08-01 06:11:00,Big Block Sing Song\n" \
                     "2020-08-01 06:13:00,Big Block Sing Song"
    filepath = "custom_dataset.csv"
    m = mock_open(read_data=custom_dataset)
    with patch("builtins.open", m, create=True):
        s = fetch_file(filepath)

    expected_index = pd.to_datetime(["2020-08-01 06:00:00", "2020-08-01 06:11:00", "2020-08-01 06:13:00"])
    expected_series = pd.Series(["The Moblees", "Big Block Sing Song", "Big Block Sing Song"],
                                index=expected_index, dtype="string", name=filepath)
    expected_series.index.name = "timestamp"

    assert_series_equal(s, expected_series)


def test_fetch_file_one_column():
    custom_dataset = "The Moblees\n" \
                     "Big Block Sing Song\n" \
                     "Big Block Sing Song"
    filepath = "custom_dataset.csv"
    m = mock_open(read_data=custom_dataset)
    with patch("builtins.open", m, create=True):
        s = fetch_file(filepath)

    expected_series = pd.Series(["The Moblees", "Big Block Sing Song", "Big Block Sing Song"],
                                dtype="string", name=filepath)
    expected_series.index.name = "timestamp"

    assert_series_equal(s, expected_series)


def test_fetch_file_two_columns_index_int():
    custom_dataset = "10,The Moblees\n" \
                     "30,Big Block Sing Song\n" \
                     "75,Big Block Sing Song"
    filepath = "custom_dataset.csv"
    m = mock_open(read_data=custom_dataset)
    with patch("builtins.open", m, create=True):
        s = fetch_file(filepath)

    expected_index = pd.Index([10, 30, 75], dtype="int64")
    expected_series = pd.Series(["The Moblees", "Big Block Sing Song", "Big Block Sing Song"],
                                index=expected_index, dtype="string", name=filepath)
    expected_series.index.name = "timestamp"

    assert_series_equal(s, expected_series)


@pytest.mark.parametrize("already_downloaded", [True, False])
def test_fetch_health_app(monkeypatch, already_downloaded):
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


# @pytest.mark.parametrize("already_downloaded", [True, False])
# def test_fetch_ubiq(monkeypatch, already_downloaded):
#     d_file = ["1_M_IS_data.dat"] if already_downloaded else []
#     if not already_downloaded:
#         monkeypatch.setattr(pd.Series, "to_csv", lambda *args, **kwargs: None)
#     monkeypatch.setattr(os, "listdir", lambda *args: d_file)
#     monkeypatch.setattr(pd, "read_csv", mock_read_csv_ubiq)
#     read_ubiq_user
#     data = fetch_ubiq()
#
#     # assert data.shape == (4,)
#     # assert isinstance(data.index, pd.DatetimeIndex)


@mock.patch('builtins.open', mock_open(
    read_data='user_info=1_M\tstart_time=2018-01-01 00:00:00\n' \
              '0\tUnlockPhone\n' \
              '60\tOpenScreen\n'))
def test_read_ubiq_user():
    filename = 'test_ISE_data.dat'
    df, user, start_time = read_ubiq_user(filename)

    assert isinstance(df, pd.Series)
    assert isinstance(user, str)
    assert isinstance(start_time, datetime)
    assert df.name == 'event'
    assert df.index.name == 'time'
    assert df.index[0] == datetime(2018, 1, 1, 0, 0, 0)
    assert df.iloc[0] == 'UnlockPhone'
    assert df.iloc[1] == 'OpenScreen'
