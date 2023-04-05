"""
Base IO for all periodic datasets
"""
import os
import re
from datetime import datetime, timedelta

import pandas as pd

from skmine.datasets._base import get_data_home

UBIQ_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00369/UbiqLog4UCI.zip"
health_app_url = "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log"
canadianTV_url = "https://zenodo.org/record/4671512/files/canadian_tv.txt"


def fetch_file(filepath, separator=',', format=None):
    """Loader for files in periodic format (timestamp,event\n). The first element can be a datetime or an integer and
    the second is a string.
    This file reader can also work for files with only one value per line (the event).
    The indexes then correspond to the line numbers.

    Parameters
    ----------
    filepath : str
        Path of the file to load

    separator : str
        Indicate a custom separator between timestamps and events. By default, it is a comma.
        If the file contains only one column, this parameter is not useful.
    format : str
        format for datetime, like "%d/%m/%Y %H:%M:%S" for day/month/year hour:min:sec
         see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior for all possibilities

    Returns
    -------
    pd.Series
        Logs from the custom dataset, as an in-memory pandas Series.
        Events are indexed by timestamps.
    """
    s = pd.read_csv(filepath, sep=separator, header=None, dtype="string", skipinitialspace=True).squeeze(axis="columns")
    if type(s) == pd.DataFrame:
        s = pd.Series(s[1].values, index=s[0])
        try:
            s.index = pd.to_datetime(s.index, format=format)
        except ValueError:
            s.index = s.index.astype("int64")
    s.index.name = "timestamp"
    s.name = filepath
    return s


def fetch_health_app(data_home=None, filename="health_app.csv"):
    """Fetch and return the health app log dataset

    see: https://github.com/logpai/loghub

    HealthApp is a mobile application for Android devices.
    Logs were collected from an Android smartphone after 10+ days of use.

    Logs have been grouped by their types, hence resulting
    in only 20 different events.

    ==============================      ===================================
    Number of occurrences               2000
    Number of events                    20
    Average delta per event             Timedelta('0 days 00:53:24.984000')
    Average nb of points per event      100.0
    ==============================      ===================================

    Parameters
    ----------
    filename : str, default: health_app.csv
        Name of the file (without the data_home directory) where the dataset will be or is already downloaded.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        System logs from the health app dataset, as an in-memory pandas Series.
        Events are indexed by timestamps.
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, index_col=0, dtype="string")
    if filename in os.listdir(data_home):
        s = pd.read_csv(p, **kwargs).squeeze(axis="columns")
    else:
        s = pd.read_csv(health_app_url, sep="|", on_bad_lines='skip', usecols=[0, 1], **kwargs).squeeze(axis="columns")
        s.to_csv(p, header=False)
    s.index.name = "timestamp"
    s.index = pd.to_datetime(s.index, format="%Y%m%d-%H:%M:%S:%f")

    return s


def fetch_canadian_tv(data_home=None, filename="canadian_tv.txt"):
    """
    Fetch and return canadian TV logs from August 2020

    see: https://zenodo.org/record/4671512

    If the dataset has never been downloaded before, it will be downloaded and stored.

    The returned dataset contains only TV series programs indexed by their associated timestamps.
    Adverts are ignored when loading the dataset.

    ==============================      =======================================
    Number of occurrences               2093
    Number of events                    98
    Average delta per event             Timedelta('19 days 02:13:36.122448979')
    Average nb of points per event      21.35714285714285
    ==============================      =======================================

    Parameters
    ----------
    filename : str, default: canadian_tv.txt
        Name of the file (without the data_home directory) where the dataset will be or is already downloaded.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets.
        By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
        TV series events from canadian TV, as an in-memory pandas Series.
        Events are indexed by timestamps.

    Notes
    -----
    For now the entire .zip file is downloaded, being ~90mb on disk
    Downloading preprocessed dataset from zenodo.org is something we consider.

    See Also
    -------
    skmine.datasets.get_data_home
    """
    data_home = data_home or get_data_home()
    p = os.path.join(data_home, filename)
    kwargs = dict(header=None, dtype="string", index_col=0)

    if filename not in os.listdir(data_home):
        s = pd.read_csv(canadianTV_url, **kwargs).squeeze(axis="columns")
        s.to_csv(p, index=True, header=False)
    else:
        s = pd.read_csv(p, **kwargs).squeeze(axis="columns")

    s.index = pd.to_datetime(s.index)
    s.index.name = "timestamp"
    s.name = "canadian_tv"
    return s

def fetch_ubiq(user_filename="25_F_ISE_data.dat", data_home=None):  # pragma : no cover
    """
    Fetch and return smartphone lifelogging event from different users
    see : https://archive.ics.uci.edu/ml/datasets/UbiqLog+%28smartphone+lifelogging%29

    If the dataset has never been downloaded before, it will be downloaded and stored.

    Parameters
    ----------
    user_filename : str, default: 1_M_IS_data.dat
       file to be loaded , by default 1_M is user , IS for normal dataset where timestamps are dropped and replace by
       1, 2, 3, 4...                                ISE for real timed dataset(events are annotated with
       Instantaneous _I, Start _S, End _E) like file 2_F_ISE_data.dat

    data_home : optional, default: None
       Specify another download and cache folder for the datasets.
       By default, all scikit-mine data is stored in `scikit-mine_data`.

    Returns
    -------
    pd.Series
       smartphone lifelogging event for the sepcified user
       Events are indexed by timestamps.

    Notes
    -----
    For now the entire .zip file is downloaded, being ~64mb on disk

    See Also
    -------
    skmine.datasets.get_data_home
    """
    data_home = data_home or get_data_home()
    ubiq_dir = os.path.join(data_home, 'UbiqLog')
    user_ubiq_dir = os.path.join(ubiq_dir, 'users_ubiq')
    if not os.path.exists(ubiq_dir):
        os.makedirs(ubiq_dir, exist_ok=True)
        os.chdir(ubiq_dir)

        infile = "all_log_applications_nonbin.txt"
        os.system("wget " + UBIQ_url)
        os.system("unzip UbiqLog4UCI.zip")
        os.system("rm __MACOSX -rf")
        os.system('grep "\\"Application\\":" UbiqLog4UCI/*/log_*.txt > ' + infile)
        os.makedirs(user_ubiq_dir)
        parse_all_user(infile, user_ubiq_dir, min_occ=10)

    filename = os.path.join(user_ubiq_dir, user_filename)
    if not os.path.exists(filename):
        raise FileNotFoundError("Searching for :" + filename)

    s, user, start_time = read_ubiq_user(filename)
    s.index.name = "timestamp"
    s.name = "Ubiq" + user

    print(f"Series loaded from {user_filename} : user {user}, start time {start_time}, nb_event {len(s)}")
    typ = "absolute time" if "ISE" in user_filename else "relative time"
    print("timestamps are in ", typ)
    return s


def read_ubiq_user(filename: str) -> tuple:
    """   Read user-event file (csv format with tabulation) and process it to return a pd.Series with event and
    timestamps as index

    Parameters
    ----------
    filename : str, default: 1_M_IS_data.dat
       file to be loaded , by default 1_M is user , IS for normal dataset (ISE if events are annotated with
       Instantaneous, Start, End)

    Returns
    -------
    tuple
        (df, user, start_time)
        df is the returned pd.Series, user is 1_M for example and , start time is the first timestamp
       smartphone lifelogging event for the specified user
       Events are indexed by timestamps.
    """
    sep = "\t"
    df = pd.read_csv(filename, sep=sep, header=None, dtype="string")
    user_info, start_time_str = df.loc[0]
    datetime_str = start_time_str.split('=')[1]
    user = user_info.split('=')[1]
    start_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    df.drop(index=df.index[0], axis=0, inplace=True)
    df.rename(columns={0: 'diff_time', 1: 'event'}, inplace=True)
    df = df.astype({"diff_time": int, "event": str})
    if filename.endswith('_IS_data.dat'):
        df['time'] = df.index  # succession of events , index = 1 2 3 4 5 ....
    elif filename.endswith('_ISE_data.dat'):
        df['time'] = df['diff_time'].apply(lambda x: start_time + timedelta(minutes=x))
        # TODO : minuts normal but no residuals , error
    else:
        raise ValueError("cant parse such files")
    df = df[['time', 'event']]
    df.set_index('time', inplace=True)
    serie = pd.Series(df['event'], index=df.index).astype('string')  # cast from object to string
    return serie, user, start_time


def parse_all_user(infile: str, out_dir: str, min_occ=10) -> None:  # pragma : no cover
    """   Parse global file with multiple user and construct csv files , one per user

    Parameters
    ----------
    infile :  str
       global event file for all users . all_log_applications_nonbin.txt by example

    out_dir: str
        directory where to write each user csv.file

    min_occ: int
        minimum of occurence : FIXME to be explained

    """
    # exemple line to parse
    # UbiqLog4UCI/10_M/log_11-29-2013.txt:{"Application":{"ProcessName":"com.broadcom.bt.app.system",
    # "Start":"11-29-2013 08:15:57","End":"11-29-2013 08:18:18"}}

    users = {}
    users_drop = set()
    with open(infile) as fp:
        for li, line in enumerate(fp):
            # print(line)
            line = '/'.join(line.strip().split('/')[1:])  # drop UbiqLog4UCI/
            # print(line)
            tmp = re.match('(?P<user>[0-9]*_[FM])/(?P<file>log_[0-9\-]+.txt):.*"ProcessName":"(?P<process>[^"]*)",'
                           '.*"Start":"(?P<start_time>[^"]*)",.*"End":"(?P<end_time>[^"]*)"', line)
            if tmp is not None:
                user = tmp.group("user")
                d = None
                if user not in users_drop:
                    try:
                        d = (datetime.strptime(tmp.group("start_time"), '%m-%d-%Y %H:%M:%S'),
                             datetime.strptime(tmp.group("end_time"), '%m-%d-%Y %H:%M:%S'))
                    except ValueError:
                        users_drop.add(user)
                        d = None

                if user not in users_drop and d is not None:
                    if user not in users:
                        users[user] = {"ev": [], "counts": {}}

                    delta = (d[1] - d[0]).total_seconds()
                    if delta < 60:  # last less than a minute
                        evs = [(d[0], "%s_I" % tmp.group("process"))]
                    else:
                        evs = [(d[0], "%s_S" % tmp.group("process")), (d[1], "%s_E" % tmp.group("process"))]
                    for (tt, ev) in evs:
                        users[user]["ev"].append((tt, ev))
                        users[user]["counts"][ev] = users[user]["counts"].get(ev, 0) + 1

    print("DROP", users_drop)
    for user, dt in users.items():
        if user not in users_drop:
            evs_tmp = [d for d in dt["ev"] if dt["counts"].get(d[1], 0) > min_occ]
            if len(evs_tmp) > min_occ:
                evs_tmp = sorted(evs_tmp)
                evs = sorted([(int((d[0] - evs_tmp[0][0]).total_seconds() / 60), d[-1]) for d in evs_tmp])
                with open("%s/%s_ISE_data.dat" % (out_dir, user), "w") as fo:
                    fo.write("### user=%s\tstart_time=%s\n" % (user, evs_tmp[0][0]))
                    prev = None
                    for pair in evs:
                        if pair != prev:
                            fo.write("%d\t%s\n" % pair)
                            prev = pair

                with open("%s/%s_IS_data.dat" % (out_dir, user), "w") as fo:
                    fo.write("### user=%s\tstart_time=%s\n" % (user, evs_tmp[0][0]))
                    prev = None
                    for tt in evs:
                        db = tt[-1].split("_")
                        if db[-1] in ["I", "S"]:
                            pair = (tt[0], "_".join(db[:-1]))
                            if pair != prev:
                                fo.write("%d\t%s\n" % pair)
                                prev = pair

