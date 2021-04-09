"""
Test MDL reconstruction for periodic patterns
"""
import pandas.testing
from skmine.periodic import PeriodicCycleMiner
from skmine.datasets import fetch_health_app, fetch_canadian_tv
from skmine.datasets.periodic import deduplicate

if __name__ == "__main__":
    Ds = [
        fetch_health_app(),
        fetch_canadian_tv(),
    ]

    for D in map(deduplicate, Ds):
        miner = PeriodicCycleMiner(max_length=20)
        print(
            f"RUN CycleMiner RECONSTRUCTION ON {D.name} WITH PARAMS {miner.get_params()}"
        )
        miner.fit(D)
        r_D = miner.reconstruct()
        pandas.testing.assert_series_equal(D, r_D, check_names=False, check_dtype=False)
