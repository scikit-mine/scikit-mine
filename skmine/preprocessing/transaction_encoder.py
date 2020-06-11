"""
Transaction encoder

This acts like
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer,
but produces pd.DataFrame as output
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class TransactionEncoder(MultiLabelBinarizer):
    __doc__ = __doc__
    def transform(self, D):
        mat = super().transform(D)
        if self.sparse_output:
            return pd.DataFrame.sparse.from_spmatrix(mat, columns=self.classes_)
        return pd.DataFrame(mat, columns=self.classes_)

    def fit_transform(self, D, y=None):
        return self.fit(D).transform(D)
