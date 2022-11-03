"""
Base IO code for all datasets
"""

import os


def get_data_home(data_home=None):
    """Return the path of the scikit-mine data home directory.

    This folder is used by some large dataset loaders to avoid downloading
    data several times.

    By default, ``data_home`` is **./scikit_mine_data/**

    Alternatively, it can be set by the **SCIKIT_MINE_DATA** environment
    variable or programmatically by giving an explicit folder path.

    Parameters
    ----------
    data_home : str | None
        The path to scikit-mine data dir.
    """
    import skmine
    sk_path = skmine.__file__  # example sk_path = '/home/username/codes/sk_root/skmine/__init__.py'
    sk_root = os.path.dirname(os.path.dirname(sk_path))  # sk_root = '/home/username/codes/sk_root

    if data_home is None:
        data_home = os.environ.get("SCIKIT_MINE_DATA", os.path.join(sk_root, "scikit_mine_data"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home
