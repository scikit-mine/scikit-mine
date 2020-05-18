import pandas as pd
from itertools import chain

from sklearn.preprocessing import MultiLabelBinarizer

class TransactionEncoder(MultiLabelBinarizer):
    def transform(self, D):
        mat = super().transform(D)
        if self.sparse_output:
            return pd.DataFrame.sparse.from_spmatrix(mat, columns=self.classes_)
        else:
            return pd.DataFrame(mat, columns=self.classes_)

    def fit_transform(self, D):
        return self.fit(D).transform(D)