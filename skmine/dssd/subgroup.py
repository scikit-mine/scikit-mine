from copy import copy
from typing import List

import pandas

from .cond import Cond
from .description import Description

class Subgroup:
    """
    A sugroup is a set of elements that match a specified description/pattern

    Parameters
    ----------
    description: Description
        The description/pattern of this subgroup
    quality: float, default=0.0
        The quality of this subgroup computed with regards to the entire dataset
    cover: pandas.Index, default=pandas.Index([])
        The indices of the subgroup members in the original dataset 
    parent: Subgroup
        This subgroup's parent. The entire dataset is a subgroup that has no parent

    Example
    -------
    >>> from skmine.dssd import Subgroup, Description, Cond
    >>> import pandas
    >>> condition1 = Cond("age", ">", 18)
    >>> pattern = Description([condition1])
    >>> sg = Subgroup(pattern, quality = .5, cover = pandas.Index([1, 5]))
    """

    def __init__(self, description: Description, quality: float = 0., cover: pandas.Index = None, parent: 'Subgroup' = None):
        self.description = description
        self.quality = quality
        self.cover = cover if cover is not None else pandas.Index([])
        self.parent = parent


    def __str__(self):
        return f"({self.description}, {len(self.cover)}, {self.quality})"


    def __repr__(self):
        return str(self)


    def __le__(self, other: 'Subgroup'):
        return self.quality <= other.quality


    def __lt__(self, other: 'Subgroup'):
        return self.quality < other.quality


    def __ge__(self, other: 'Subgroup'):
        return self.quality >= other.quality


    def __gt__(self, other: 'Subgroup'):
        return self.quality > other.quality


    def __eq__(self, other: 'Subgroup'):
        return id(self) == id(other) or self.description == other.description


    def __ne__(self, other: 'Subgroup'):
        return self.description != other.description


    def __hash__(self):
        return hash(tuple(self.description.conditions))


    def child_with_new_condition(self, new_cond: Cond):
        """
        Create a new subgroup that has this subgroup as its parent, 
        same conditions plus the new condition received in parameter,
        an empty cover and a quality of 0.0.
        NB: This method does NOT compute the cover for the child subgroup
        and the conditions are copied by reference (they are neither cloned nor deep copied).

        Parameters
        ----------
        new_cond: Cond
            The new condition to add to the resulting child subgroup

        Returns
        -------
        Subgroup
        """
        return Subgroup(Description([*self.description.conditions, new_cond]), quality=0, cover=None, parent=self)
