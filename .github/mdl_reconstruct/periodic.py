from skmine.periodic import PeriodicCycleMiner
from skmine.datasets import fetch_health_app
import pandas as pd

if __name__ == "__main__":
    Ds = [fetch_health_app()]

    miners = [
        PeriodicCycleMiner(),
        # SLIM(pruning=True, n_iter_no_change=1000)
    ]
    for D in Ds:
        _D = D.groupby(D.index).first()
        for miner in miners:
            print(
                f"RUN {type(miner)} RECONSTRUCTION ON {D.name} WITH PARAMS {miner.get_params()}"
            )
            miner.fit(D)
            r_D = miner.reconstruct()
            pd.testing.assert_series_equal(_D, r_D, check_names=False, check_dtype=False)
