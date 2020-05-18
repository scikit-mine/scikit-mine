"""Base classes for all miners."""

from abc import ABC
from abc import abstractmethod

from collections import defaultdict
import inspect


class BaseMiner(ABC):
    """Base class for all miners in scikit-mine."""

    @abstractmethod
    def fit(self, D, y=None):
        """Fit method to be implemented."""
        pass


    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self):
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
        if not params: return self

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            setattr(self, key, value)
            valid_params[key] = value

        return self


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
