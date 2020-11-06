from skmine.itemsets import SLIM
from skmine.datasets import make_transactions
from skmine.preprocessing import TransactionEncoder


class SLIMBench:
    params = ([20, 1000], [0.3, 0.7])
    param_names = ["n_transactions", "density"]
    # timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1

    def setup(self, n_transactions, density):
        transactions = make_transactions(
            n_transactions=n_transactions,
            density=density,
            random_state=7,
        )
        self.transactions = TransactionEncoder().fit_transform(transactions)
        self.slim = SLIM()

    def time_fit(self, *args):
        self.slim.fit(self.transactions)

    def track_data_size(self, *args):
        return self.slim.fit(self.transactions).data_size_
