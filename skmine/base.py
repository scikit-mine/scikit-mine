"""Base classes for all miners."""
# pylint: disable= unused-argument

import inspect
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


class BaseMiner(BaseEstimator):
    """Base class for all miners in scikit-mine compatible with BaseEstimator from scikit-learn"""

    @abstractmethod
    def fit(self, D, y=None):
        """Fit method to be implemented."""
        return self

    @abstractmethod
    def discover(self, *args, **kwargs):
        """discover method to be implemented."""
        return pd.DataFrame()

    def fit_discover(self, D, y=None, **kwargs):
        """
        Fit to data, the extract patterns

        Parameters
        ----------
        D: {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)

        Returns
        -------
        pd.Dataframe
            patterns discovered by a mining algorithm
        """
        if y is None:
            return self.fit(D).discover(**kwargs)
        return self.fit(D, y=y).discover(**kwargs)


class TransformerMixin(TransformerMixin):
    """Base Mixin for transformers in scikit-mine compatible with TransformerMixin from scikit-learn"""


class MDLOptimizer(ABC):
    """
    Base interface for all models applying the `Minimum Description Length principle
    <https://en.wikipedia.org/wiki/Minimum_description_length>`_.
    """

    @abstractmethod
    def generate_candidates(self, *args, **kwargs):
        """
        Generate new candidates, to be sent for later evaluation.

        Calling this function is equivalent to sending a new message given an encoding scheme,
        while calling ``.evaluate`` is equivalent to receiving this message, and evaluating the gain
        of information it provides.

        Returns
        -------
        object or Iterable[object]
            A set of new candidates
        """
        return list()

    @abstractmethod
    def evaluate(self, candidate, *args, **kwargs):
        """
        Evaluate the gain, i.e the gain of information when accepting the candidate.

        Parameters
        ----------
        candidate: object
            A candidate to evaluate

        Returns
        -------
        tuple (data_size, model_size, ...)
            Should return a tuple, with first two values corresponding to new data size
            and model size in the case of accepting the candidate.

            Data size and model size should be returned separately as we encourage
            usage of `(two-part) crude MDL
            <https://en.wikipedia.org/wiki/Minimum_description_length#Two-Part_Codes>`_.
        """
        return (
            0,
            0,
        )

    def _repr_html_(self):
        df = self.discover()  # call discover with default parameters
        if not df.empty:
            return df._repr_html_()  # pylint: disable=protected-access
        return repr(self)


class InteractiveMiner(ABC):
    """Base class for interactive mining

    Interactive miners should allow us to
    1. ingest some input data, by calling `prefit`
    2. generate candidates
    3. loop over generated candidate, and call `update` with this candidate as argument,
    depending on some external input (like a positive answer from a user in CLI mode)
    """

    @abstractmethod
    def prefit(self, D):
        """ingest data `D` and track basic informations to be used later"""
        return self

    @abstractmethod
    def update(self, *args, **kwargs):
        """inplace edition of underlying datastructures"""
        return None
