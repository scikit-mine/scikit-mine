from typing import Any


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
        self.__tuple = (attribute, op, val)
        self.__attr_str = f"`{attribute}`" if ' ' in attribute else attribute
        self.__val_str = str(val) if isinstance(val, (int, float,)) else f'"{val}"' if isinstance(val, str) else val # bool cases included


    @property
    def attribute(self): return self.__tuple[0]

    @property
    def op(self): return self.__tuple[1]

    @property
    def val(self): return self.__tuple[2]

    def __eq__(self, other: 'Cond'):
        return self.__tuple == other.__tuple
        return self.attribute == other.attribute and self.op == other.op and self.val == other.val


    def __str__(self):
        """Return a pandas like string that can be used to query a pandas DataFrame by using the df.query() method"""
        return f"({self.__attr_str} {self.op} {self.__val_str})"


    def __repr__(self):
        return str(self)
    

    # method mainly used for removing duplicates of patterns (list of conditions)
    def __hash__(self):
        return hash(self.__tuple)
