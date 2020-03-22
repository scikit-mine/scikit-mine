"""Base classes for all miners."""

from abc import ABC
from abc import abstractmethod

import numpy as numpy
import pandas as pd


class BaseMiner(ABC):
    """Base class for all miners in scikit-mine."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit method to be implemented."""
        pass


class MDLOptimizer(ABC):
    """
    Base interface for all models applying the Minimum Description Length principle.

    see: https://en.wikipedia.org/wiki/Minimum_description_length
    """

    @abstractmethod
    def evaluate_gain(self, *args, **kwargs):
        """
        Evaluate the gain, i.e compute L(D|CT) - L(CT|D).

        L(D|CT) - L(CT|D) is the difference between 
        the size of the dataset D encoded with the codetable CT and the size of the codetable CT
        """
        pass

    @abstractmethod
    def cover(self, codetable, D):
        """Cover the dataset D given the codetable."""
        pass



class TransformerMixin:
    """Mixin class for all transformers in scikit-mine."""

    def fit_transform(self, D, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to D with optional parameters fit_params
        and returns a transformed version of D.
        Parameters
        ----------
        D : pd.Series
            A set of transactions
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        D_new : pd.Series
            Transformed transactions
        """
        return self.fit(D, **fit_params).transform(D)
