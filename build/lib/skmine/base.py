"""Base classes for all miners."""

from abc import ABC
from abc import abstractmethod


class BaseMiner(ABC):
    """Base class for all miners in scikit-mine."""

    @abstractmethod
    def fit(self, D, y=None):
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
