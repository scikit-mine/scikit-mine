from typing import Any


class Cond:
    """
    A class representing a condition
    
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


    def __eq__(self, other: 'Cond'):
        return self.attribute == other.attribute and self.op == other.op and self.val == other.val


    def __str__(self):
        """Return a pandas like string that can be used to query a pandas DataFrame by using the df.query() method"""
        if isinstance(self.val, (int, float,)):
            val_str = str(self.val)
        elif isinstance(self.val, str):
            val_str = f'"{self.val}"'
        else: # bool cases included
            val_str = self.val
        return f"({self.attribute} {self.op} {val_str})"


    def __repr__(self):
        return str(self)
    

    # method mainly used for removing duplicates of patterns (list of conditions)
    def __hash__(self):
        return hash((self.attribute, self.op, self.val))
