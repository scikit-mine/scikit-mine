"""
Test MDL reconstruction for periodic patterns
"""
import pandas.testing
import pandas as pd
from skmine.periodic import PeriodicPatternMiner
from skmine.datasets import fetch_health_app, fetch_canadian_tv

if __name__ == "__main__":
    Ds = [fetch_canadian_tv(), fetch_health_app()]

    for D in Ds:
        if D.index.duplicated().any():  # there are potentially duplicates, i.e. occurrences that happened at the
            # same time AND with the same event. At this line, the second condition is not yet verified.
            D_no_duplicate = D.groupby(by=D.index).apply(lambda x: x.drop_duplicates())
            # if same time and same event,  create Multi inde names =[timestamp, timestamp]
            D_no_duplicate = D_no_duplicate.reset_index(level=0, drop=True)
        else:
            D_no_duplicate = D

        miner = PeriodicPatternMiner()
        print(f"RUN CycleMiner RECONSTRUCTION ON {D.name} WITH PARAMS {miner.get_params()}")
        miner.fit(D)
        recons_from_pattern = miner.reconstruct()
        residuals = miner.get_residuals()
        recons = pd.concat([residuals, recons_from_pattern], ignore_index=True)
        recons.set_index('time', inplace=True)
        recons.sort_index(inplace=True)
        recons = recons.squeeze()  # cast from Dataframe to Series
        recons.index.name = 'timestamp'
        # if same timestamp , sort series in alphabetic ordrer
        recons = recons.sort_values().sort_index(kind='mergesort')
        D_no_duplicate = D_no_duplicate.sort_values().sort_index(kind='mergesort')

        pandas.testing.assert_series_equal(D_no_duplicate, recons, check_names=False, check_dtype=False)
