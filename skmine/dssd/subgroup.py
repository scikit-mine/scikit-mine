from copy import copy

import pandas

from .cond import Cond
from .description import Description

class Subgroup:
    """A candidate is a set of elements that match a specified description/pattern

    Attributes:
    -----------
    * description (Description): the description/pattern of this candidate subgroups
    * quality (float): the quality of this candidate computed with the parameters set on this instance of the algorithm
    * cover (pandas.Index): the indices of the candidate members in the original dataset 
    * parent (Candidate): this candidate's parent 
    """

    def __init__(self, description: Description, quality: float = 0., cover: pandas.Index = pandas.Index([]), parent: 'Subgroup' = None):
        self.description = description
        self.quality = quality
        self.cover: pandas.Index = cover
        self.parent = parent


    def __str__(self):
        return f"({self.description}, {len(self.cover)}, {self.quality})"


    def __repr__(self):
        return str(self)


    # A single rich operator could've been implemented but to avoid possible
    # performance penalties, we implemented these manually instead of using
    # something like functools
    def __le__(self, other: 'Subgroup'):
        return self.quality <= other.quality


    def __lt__(self, other: 'Subgroup'):
        return self.quality < other.quality


    def __ge__(self, other: 'Subgroup'):
        return self.quality >= other.quality


    def __gt__(self, other: 'Subgroup'):
        return self.quality > other.quality


    def __eq__(self, other: 'Subgroup'):
        return self.description == other.description


    def __ne__(self, other: 'Subgroup'):
        return self.description != other.description


    def __hash__(self):
        return hash(tuple(self.description.conditions))


    def clone(self) -> 'Subgroup':
        return copy(self)


    def child_with_new_condition(self, new_cond: Cond):
        """Create a new candidate with the following characteristics. It has:
        * all this candidate's conditions plus the new condition
        * this candidate as its parent
        * an empty cover
        * a zero quality"""
        return Subgroup(Description([*self.description.conditions, new_cond]), quality=0, cover=None, parent=self)
