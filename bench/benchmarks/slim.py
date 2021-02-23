from skmine.itemsets import SLIM
from skmine.datasets import make_transactions
from sklearn.preprocessing import MultiLabelBinarizer
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
        mlb = MultiLabelBinarizer()
        self.transactions = mlb.fit_transform(transactions)
        self.new_transaction = mlb.transform(transactions.sample(len(transactions)))
        self.fitted_slim = SLIM().fit(transactions)

    def time_fit(self, *args):
        SLIM().fit(self.transactions)

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

    def mem_fit(self, *args):
        SLIM().fit(self.dataset)
