from skmine.datasets import make_transactions
from skmine.itemsets import LCM
import numpy as np

class TimeLCM:
    params = ([.1, .3, .5], [200], [20, 50], [10, 30])
    param_names = ['min_supp', 'n_transactions', 'n_items', 'avg_transaction_size']
    #timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1
    def setup(self, min_supp, n_transactions, n_items, avg_transaction_size):
        np.random.seed(7)
        self.transactions = make_transactions(
            n_transactions,
            n_items,
            avg_transaction_size,
        )
        self.lcm = LCM(min_supp=min_supp, n_jobs=1)  # set n_jobs to 1 for now

    def time_fit(self, *args):
        self.lcm.fit(self.transactions)

    def time_fit_transform(self, *args):
        self.lcm.fit_transform(self.transactions)
