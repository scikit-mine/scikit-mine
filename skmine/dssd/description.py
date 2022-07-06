
from typing import List
from .cond import Cond


class Description:
    """
    
    Attributes:
    -----------

    conditions: list[Cond]
    op: str
        a string value representing the logic operator & or | used to link together the conditions.
        It default to "&" which is what is proposed in the dssd paper.
        It is the only operator that makes sense here
    """
    def __init__(self, conditions: List[Cond] = None, op: str = "&"):
        """_summary_

        Args:
            conditions (list[Cond], optional): conditions to set on this pattern/description. Defaults to [].
            op (str, optional): logic operator between conditions in. Defaults to "&".
        """
        self.conditions = conditions or []
        self.op = op


    def __len__(self):
        return len(self.conditions)


    def __str__(self):
        """Return a string that can be used to query a pandas dataframe using the df.query() method"""
        return f" {self.op} ".join(str(cond) for cond in self.conditions)


    def __eq__(self, other: 'Description'):
        return self.op == other.op and set(self.conditions) == set(other.conditions)
        

    def __contains__(self, cond: Cond):
        return cond in self.conditions


    def is_attribute_used(self, attr: str) -> bool:
        """Checks if any of the conditions use the the specified attribute"""
        return any(c.attribute == attr for c in self.conditions)


    def has_equal_condition_on_attr(self, attr: str) -> bool:
        """Checks if any of the conditions on the specified attribute is an EQUAL one"""
        return any(c.attribute == attr and c.op == "==" for c in self.conditions)
