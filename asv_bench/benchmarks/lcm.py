from skmine.itemsets import LCM
from skmine.datasets import make_transactions

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
            random_state=7
        )
        self.lcm = LCM(min_supp=min_supp, n_jobs=1)  # set n_jobs to 1 for now

    def time_fit_only(self, *args):
        self.lcm.fit(self.transactions)

    def time_fit_transform(self, *args):
        self.lcm.fit_transform(self.transactions)
