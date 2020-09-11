""" Transaction Encoder"""
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from ..bitmaps import Bitmap


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
    """`TransactionEncoder` acts like `sklearn's MultiLabelBinarizer
    <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer>`_,
    but differs in many ways:

    - it produces a pandas.DataFrame as output, either dense or sparse,
      depending on the ``sparse_output`` argument.
    - the number of "columns" can vary from one call to ``.transform`` to another.
    - input is possibly out of core : works on generators.


    Parameters
    ----------
    sparse_output: bool
        True if a sparse output is to be produced.
    """
    def __init__(self, sparse_output=True):
        self.sparse_output = sparse_output

    def fit(self, D): # pylint: disable=missing-function-docstring
        if not isinstance(D, Iterable):
            raise TypeError('D should be a list of list, or at least an iterator of iterator')
        return self

    partial_fit = fit

    def transform(self, D):
        """Apply transformation on the transactional input

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

        Examples
        --------
        >>> from skmine.preprocessing import TransactionEncoder
        >>> transactions = [['banana', 'milk'], ['milk', 'cookies', 'banana']]
        >>> te = TransactionEncoder()
        >>> te.fit_transform(transactions)
           banana  cookies  milk
        0    True    False  True
        1    True     True  True

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

        return pd.DataFrame(mat, columns=cols)

    def fit_transform(self, D):
        """Fit then transform on transactional data"""
        return self.fit(D).transform(D)
