from skmine.itemsets import SLIM
from skmine.datasets import make_transactions

class SLIMBench:
    params = ([20], [.4, .6], [False, True])
    param_names = ['n_transactions', 'density', 'pruning']
    #timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1
    def setup(self, n_transactions, density, pruning):
        self.transactions = make_transactions(
            n_transactions=n_transactions,
            density=density,
            random_state=7,
        )
        self.slim = SLIM(pruning=pruning)

    def time_fit(self, *args):
        self.slim.fit(self.transactions)

    def mem_fit(self, *args):
        self.slim.fit(self.transactions)

    def track_data_size(self, *args):
        return self.slim.fit(self.transactions).data_size
