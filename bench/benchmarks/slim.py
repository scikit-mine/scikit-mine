from skmine.itemsets import SLIM
from skmine.datasets import make_transactions
from skmine.datasets.fimi import fetch_any


class SLIMBench:
    params = ([20, 1000], [100, 500])
    param_names = ["n_transactions", "n_items"]
    # timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1

    def setup(self, n_transactions, n_items):
        transactions = make_transactions(
            n_transactions=n_transactions, n_items=n_items, random_state=7,
        )
        new_transaction = transactions.sample(len(transactions))
        self.new_transactions = (
            new_transaction.map(str).str.join("|").str.get_dummies(sep="|")
        )  # tabular
        self.fitted_slim = SLIM().fit(transactions)
        self.transactions = transactions

    def time_fit(self, *args):
        SLIM().fit(self.transactions)

    def time_prefit(self, *args):
        s = SLIM()
        if hasattr(s, "_prefit"):
            s._prefit(self.transactions)
        else:
            s.prefit(self.transactions)

    def time_decision_function(self, *args):
        self.fitted_slim.decision_function(self.new_transactions)

    def track_data_size(self, *args):
        return self.fitted_slim.data_size_

    def track_codetable_size(self, *args):
        return sum(map(lambda iset: len(iset) > 1, self.fitted_slim.codetable_))


class SLIMStandardDatasets:
    params = ["chess.dat", "kosarak.dat", "mushroom.dat"]
    param_names = ["dataset"]

    def setup(self, dataset):
        self.data = fetch_any(dataset)

    def time_fit(self, *args):
        SLIM().fit(self.data)

    def peakmem_fit(self, *args):
        SLIM().fit(self.data)
