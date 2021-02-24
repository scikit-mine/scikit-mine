from skmine.periodic import PeriodicCycleMiner
from skmine.datasets.periodic import fetch_health_app


class PeriodicCycleBench:
    params = ([10, 100, 200])
    param_names = ["max_length"]
    # timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1

    def setup(self, max_length):
        self.logs = fetch_health_app()
        self.miner = PeriodicCycleMiner(max_length=max_length)
        self.max_length = max_length

    def time_fit(self, max_length):
        PeriodicCycleMiner(n_jobs=1, max_length=max_length).fit(self.logs)

    def mem_fit(self, max_length):
        PeriodicCycleMiner(n_jobs=1, max_length=max_length).fit(self.logs)
