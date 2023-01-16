"""Base classes for all miners."""
# pylint: disable= unused-argument

import inspect
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseMiner(ABC):
    """Base class for all miners in scikit-mine."""

    @abstractmethod
    def fit(self, D, y):
        """Fit method to be implemented."""
        return self

    @abstractmethod
    def discover(self, *args, **kwargs):
        """discover method to be implemented."""
        return pd.Series()

    def _get_tags(self):
        return {
            "non_deterministic": False,
            "requires_positive_X": True,
            "requires_positive_y": False,
            "X_types": ['2darray'] , #["categorical"],
            "poor_score": False,
            "no_validation": False,
            "multioutput": False,
            "allow_nan": False,
            "stateless": True,
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
        parameters = [p for p in init_signature.parameters.values()
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
                raise ValueError("Invalid parameter %s for estimator %s. Check the list of available parameters "
                                 "with `estimator.get_params().keys()`." % (key, self))
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
        else:
            return self.fit(D, y=y).discover(**kwargs)


class TransformerMixin:
    """Base Mixin for transformers in scikit-mine"""

    def fit_transform(self, X, y=None):
        """fit on X and y, then transform X"""
        return self.fit(X, y).transform(X)

    decision_function = None


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
        return (0, 0,)

    def _repr_html_(self):
        s = self.discover()  # call discover with default parameters
        df = s.to_frame(name="usage")
        if not df.empty:
            return df._repr_html_()  # pylint: disable=protected-access
        else:
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
