from skmine.itemsets import LCM
import numpy as np
import pandas as pd
import warnings

def make_transactions(n_transactions, n_items, avg_transaction_size, seed=None):
    if avg_transaction_size > n_items:
        warnings.warn(
            'Average transaction size bigger than n_items'
        )

    choices = np.arange(n_items)
    np.random.seed(seed)
    t_sizes = np.random.binomial(
        n=min(avg_transaction_size * 2, n_items),  # avoid more than n_items in single trans
        p=.5,  # centered around avg_transaction_size
        size=n_transactions
    )

    D = [np.random.choice(choices, size, replace=False) for size in t_sizes]
    return pd.Series(D)

class TimeLCM:
    params = ([.1, .3, .5], [20], [50, 100])
    param_names = ['min_supp', 'n_transactions', 'n_items']
    #timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 10.0)
    processes = 1
    def setup(self, min_supp, n_transactions, n_items):
        avg_transaction_size = n_items // 2
        self.transactions = make_transactions(
            n_transactions,
            n_items,
            avg_transaction_size,
            seed=7
        )
        self.lcm = LCM(min_supp=min_supp, n_jobs=1)  # set n_jobs to 1 for now

    def time_fit(self, *args):
        self.lcm.fit(self.transactions)

    def time_fit_transform(self, *args):
        self.lcm.fit_transform(self.transactions)
