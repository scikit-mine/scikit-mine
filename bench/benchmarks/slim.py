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
        te = TransactionEncoder()
        self.transactions = te.fit_transform(transactions)
        self.new_transactions = te.transform(transactions.sample(len(transactions)))
        self.slim = SLIM()
        self.fitted_slim = SLIM().fit(self.transactions)

    def time_fit(self, *args):
        self.slim.fit(self.transactions)

    def time_decision_function(self, *args):
        self.fitted_slim.decision_function(self.new_transactions)

    def track_data_size(self, *args):
        return self.slim.data_size_
