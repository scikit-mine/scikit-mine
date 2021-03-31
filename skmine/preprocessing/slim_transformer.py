import warnings
from itertools import islice

import numpy as np
import pandas as pd

from ..base import BaseMiner, TransformerMixin, _get_tags
from ..itemsets import SLIM
from ..itemsets.slim import _log2, _to_vertical, cover

STRATEGIES = ("codes", "one-hot")


class SLIMTransformer(SLIM, TransformerMixin):
    """SLIM mining, turned into a preprocessing step for sklearn

    `k` new itemsets (associations of one or more items) are learned at training time
    The model (pattern set) is then applied to new data by covering it (in standard cover order).
    This is similar to one-hot-encoding, except the dimension will be much more concise,
    because the columns will be patterns learned via an MDL criterion.

    If the chosen strategy is set to `one-hot`, non-zero cells are filled with ones
    If the chosen `strategy` is left to `codes`, non-zero cells are filled with code length,
    i.e the probabities of the pattern in the training data.

    See Also
    --------
    skmine.itemsets.SLIM

    Notes
    -----
    This transformer does not output scipy.sparse matrices,
    as SLIM should learn a concisedescription of the data,
    and covering new data with this small set of high usage
    patterns should output matrices with very few zeros.
    """

    decision_function = None

    def __init__(self, strategy="codes", *, k=3, **kwargs):
        self.k = k
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy = strategy
        SLIM.__init__(self, **kwargs)

    def fit(self, D, y=None):
        self._prefit(D)  # TODO : pass y ?
        seen_cands = set()
        # if self.k > len(self.standard_codetable_):
        #    warnings.warn(f"k parameter bigger than number of single items in data")
        while (len(self.codetable_) - len(self.standard_codetable_)) <= self.k:
            candidates = self.generate_candidates(stack=seen_cands)
            for cand, _ in candidates:
                data_size, model_size, update_d, prune_set = self.evaluate(cand)
                diff = (self.model_size_ + self.data_size_) - (data_size + model_size)

                if diff > 0.01:  # underflow
                    self.codetable_.update(update_d)
                    if self.pruning:
                        self.codetable_, data_size, model_size = self._prune(
                            self.codetable_, prune_set, model_size, data_size
                        )

                    self.data_size_ = data_size
                    self.model_size_ = model_size
            if not candidates:  # if empty candidate generation
                print(f"could not discover {self.k} itemsets. Early stopped")
                break
        return self

    def transform(self, D, y=None):
        """Transform new data

        Parameters
        ----------
        D: iterable
            transactional data

        Returns
        -------
        pd.DataFrame
            a dataframe of `len(D)` rows and `self.k` columns

        See Also
        --------
        skmine.itemsets.SLIM.cover
        """
        mat = self.cover(D)  # start by covering
        isets = self.codetable_.keys()[: self.k]
        mat = mat.loc[:, isets]

        if self.strategy == "codes":
            codetable = pd.Series(self.codetable_, dtype=object)
            code_lengths = codetable.map(
                len
            )  # TODO : keep codelength as internal attribute ?
            ct_codes = code_lengths[isets] / code_lengths.sum()
            codes = (mat * ct_codes).astype(np.float32)
            mat = -(np.log2(codes.replace(0.0, np.nan)).replace(np.nan, 0.0))

        return mat

    def _get_tags(self):
        return {**_get_tags(self), **{"stateless": True}}  # modfies output shape
