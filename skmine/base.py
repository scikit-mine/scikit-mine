"""Base classes for all miners."""
# pylint: disable= unused-argument

import inspect
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


def _get_tags(self):
    return {
        "non_deterministic": False,
        "requires_positive_X": False,
        "requires_positive_y": False,
        "X_types": ["2darray"],
        "poor_score": False,
        "no_validation": True,
        "multioutput": False,
        "allow_nan": False,
        "stateless": False,
        "multilabel": False,
        "_skip_test": False,
        "_xfail_checks": False,
        "multioutput_only": False,
        "binary_only": False,
        "requires_fit": True,
        "preserves_dtype": [np.float64],
        "requires_y": False,
        "pairwise": False,
    }


class BaseMiner(ABC):
    """Base class for all miners in scikit-mine."""

    @abstractmethod
    def fit(self, D, y):
        """Fit method to be implemented."""
        return self

    _get_tags = _get_tags

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=False):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            setattr(self, key, value)
            valid_params[key] = value

        return self


class DiscovererMixin:
    """Mixin for all pattern discovery models in scikit-mine"""

    def fit_discover(self, D, y=None, **kwargs):
        """
        Fit to data, the extract patterns

        Parameters
        ----------
        D: {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)

        Returns
        -------
        pd.Series
            patterns discovered by a mining algorithm
        """
        if y is None:
            return self.fit(D).discover(**kwargs)
        return self.fit(D, y=y).discover(**kwargs)


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

    @property
    def codetable(self):
        """Get a user-friendly copy of the codetable

        Returns
        -------
        pd.Series
            codetable containing patterns and ids of transactions in which they are used
        """
        ct = getattr(self, "codetable_", None)
        if ct is not None:
            l = {
                iset: tids.copy()
                for iset, tids in self.codetable_.items()
                if len(tids) > 0
            }
            return pd.Series(l, dtype="object")
        raise NotImplementedError()

    def _repr_html_(self):
        ct = getattr(self, "codetable", None)
        if ct is not None:
            if isinstance(ct, pd.Series):
                df = ct.to_frame(name="usage")
                return df._repr_html_()  # pylint: disable=protected-access
            return repr(ct)
        return repr(self)
