"""
Transaction encoder

This acts like
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer,
but differs in many ways:
    - it produces a pandas.DataFrame as output, either dense or sparse,
    depending on the ``sparse_output`` argument
    - the number of "columns" can vary. Unlike in scikit-learn, there is no need to freeze fit
    some input and make new input fit this definition.
    - input is possible out of core : works on generators.
"""
import pandas as pd
import numpy as np
from ..bitmaps import Bitmap
from sortedcontainers import SortedSet
from collections import defaultdict

from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

def make_vertical(D: pd.Series):
    """
    Applied on an original dataset this makes up a standard codetable
    """
    codetable = defaultdict(Bitmap)
    for idx, transaction in enumerate(D):
        for item in transaction:
            codetable[item].add(idx)
    return pd.Series(codetable)


class TransactionEncoder():
    __doc__ = __doc__
    def __init__(self, sparse_output=True):
        self.sparse_output = sparse_output

    def fit(self, D): return self  # just for compat and usage  pylint: disable=missing-function-docstring

    def transform(self, D):
        """
        Apply transformation on the transactional input

        Parameters
        ----------
        D: iterable of iterable
            A transactional dataset, in the form of a list of list, or a generator

        Returns
        -------
        pandas.DataFrame
            a DataFrame of boolean values. A cell contains a boolean values stating if the item
            corresponding to the column was present in the transaction corresponding to the row.

            if sparse_output is True, each column will be stored as a pandas.arrays.SparseArray
        """
        vert = defaultdict(Bitmap)
        n_transactions = 0
        for transaction in D:
            for item in transaction:
                vert[item].add(n_transactions)
            n_transactions += 1

        shape = (n_transactions, len(vert))
        if self.sparse_output:
            mat = lil_matrix(shape, dtype=bool)
        else:
            mat = np.zeros(shape, dtype=bool)

        cols = sorted(vert.keys())
        for col_idx, col in enumerate(cols):
            tids = vert[col]
            mat[tids, col_idx] = True


        if self.sparse_output:
            return pd.DataFrame.sparse.from_spmatrix(mat, columns=cols)
        else:
            return pd.DataFrame(mat, columns=cols)

    def fit_transform(self, D):
        return self.fit(D).transform(D)
