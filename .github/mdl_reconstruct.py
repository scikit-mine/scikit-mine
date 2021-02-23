"""
Python script to ensure full reconstruction of datasets using MDL miners

As MDL is a lossless compression framework, the entire original data should be reconstructed from
the concise representation that MDL provides
"""

import pandas as pd
from skmine.itemsets import SLIM
from skmine.datasets.fimi import fetch_any


if __name__ == "__main__":
    Ds = [fetch_any(k) for k in ("chess.dat", "connect.dat", "mushroom.dat")]

    miners = [
        SLIM(pruning=False, n_iter_no_change=100),
        SLIM(pruning=True, n_iter_no_change=100)
    ]
    for D in Ds:
        for miner in miners:
            print(
                f"RUN {type(miner)} RECONSTRUCTION ON {D.name} WITH PARAMS {miner.get_params()}"
            )
            miner.fit(D)
            r_D = miner.reconstruct()
            pd.testing.assert_series_equal(D, r_D, check_names=False)
