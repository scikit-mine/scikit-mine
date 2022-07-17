
from typing import List
from .cond import Cond


class Description:
    """
    A subgroup description that consists of conditions on the descriptive attributes

    Parameters
    ----------
    conditions: List[Cond], default=[]
        The list of conditions in the description

    Examples
    --------
    >>> from skmine.dssd import Description, Cond
    >>> pattern = Description([Cond("age", ">", 18), Cond("age", "<", 20)])
    >>> pattern
    (age > 18) & (age < 20)
    """

    def __init__(self, conditions: List[Cond] = None):
        self.conditions = conditions or []


    def __len__(self):
        return len(self.conditions)


    def __str__(self):
        """Return a string that can be used to query a pandas dataframe using the df.query() method"""
        return f" & ".join(str(cond) for cond in self.conditions)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: 'Description'):
        return id(self) == id(other) or set(self.conditions) == set(other.conditions)
        

    def __contains__(self, cond: Cond):
        return cond in self.conditions


    def is_attribute_used(self, attr: str) -> bool:
        """
        Checks if any of the conditions in this description use the the specified attribute
        
        Parameters
        ----------
        attr: str
            The name of the attribute to look for in this description's conditions

        Returns
        -------
        bool
        """
        return any(c.attribute == attr for c in self.conditions)


    def has_equal_condition_on_attr(self, attr: str) -> bool:
        """
        Checks if any of the conditions in this description on 
        the specified is an equal condition is an equal condition

        Parameters
        ----------
        attr: str
            The name of the attribute to look for in this description's conditions

        Returns
        -------
        bool
        """
        return any(c.attribute == attr and c.op == "==" for c in self.conditions)
