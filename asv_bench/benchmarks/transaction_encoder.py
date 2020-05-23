from skmine.preprocessing import TransactionEncoder
from skmine.datasets import make_transactions

class TransactionEncoderBench:
    params = ([500, 2000], [.2, .8], [False, True])
    param_names = ['n_transactions', 'density', 'sparse_output']
    #timeout = 20  # timeout for a single run, in seconds
    repeat = (1, 5, 20.0)
    processes = 1
    def setup(self, n_transactions, density, sparse_output):
        self.transactions = make_transactions(
            n_transactions=n_transactions,
            density=density,
            random_state=7,
        )
        self.te = TransactionEncoder(sparse_output=sparse_output)

    def time_fit_transform(self, *args):
        return self.te.fit_transform(self.transactions)

    def mem_fit(self, *args):
        return self.te.fit_transform(self.transactions)
