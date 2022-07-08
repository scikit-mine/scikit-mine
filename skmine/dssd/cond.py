from typing import Any

from attr import attr


class Cond:
    """
    A class representing a condition that should not be mutated once created
    
    Parameters
    ----------
    attribute: str: 
        The left operand of the condition as an attribute's name 
        that can later be used to select data from a dataset
    op: str, any of these: <, <=, ==, !=, >, >=
        The comparison operator
    val: Any 
        The actual value to be used as the right operand when 
        evaluating the conditions against a dataset the condition

    Examples
    --------
    >>> from skmine.dssd import Cond
    >>> condition1 = Cond("age", ">", 18)
    """

    def __init__(self, attribute: str, op: str, val: Any):
        self.attribute = attribute
        self.op = op
        self.val = val
        self.__attr_str = f"`{self.attribute}`" if ' ' in self.attribute else self.attribute
        if isinstance(self.val, (int, float,)):
            self.__val_str = str(self.val)
        elif isinstance(self.val, str):
            self.__val_str = f'"{self.val}"'
        else: # bool cases included
            self.__val_str = self.val


    def __eq__(self, other: 'Cond'):
        return self.attribute == other.attribute and self.op == other.op and self.val == other.val


    def __str__(self):
        """Return a pandas like string that can be used to query a pandas DataFrame by using the df.query() method"""
        return f"({self.__attr_str} {self.op} {self.__val_str})"


    def __repr__(self):
        return str(self)
    

    # method mainly used for removing duplicates of patterns (list of conditions)
    def __hash__(self):
        return hash((self.attribute, self.op, self.val))
