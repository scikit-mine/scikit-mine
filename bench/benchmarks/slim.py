from skmine.itemsets import SLIM
from skmine.datasets import make_transactions
from skmine.datasets.fimi import fetch_any


class SLIMBench:
    params = ([20, 1000], [0.3, 0.7])
    param_names = ["n_transactions", "density"]
    # timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 3, 20.0)
    processes = 1

    def setup(self, n_transactions, density):
        transactions = make_transactions(
            n_transactions=n_transactions, density=density, random_state=7,
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
        SLIM()._prefit(self.transactions)

    def time_decision_function(self, *args):
        self.fitted_slim.decision_function(self.new_transactions)

    def track_data_size(self, *args):
        return self.fitted_slim.data_size_


class SLIMStandardDatasets:
    params = ["chess", "kosarak", "mushroom"]
    param_names = ["dataset"]

    def setup(self, dataset):
        self.dataset = dataset

    def time_fit(self, *args):
        SLIM().fit(self.dataset)

    def peakmem_fit(self, *args):
        SLIM().fit(self.dataset)
