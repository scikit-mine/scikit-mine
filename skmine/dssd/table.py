from typing import Dict
from pandas import DataFrame
import pandas

from .custom_types import ColumnType

class Table:
    """
    A tiny little wrapper around pandas dataframe to hold potentially 
    usefull information regarding the dssd task

    Parameters
    ----------
    df: DataFrame
        Inner pandas dataframe
    column_types (dict[str, ColumnType]): 
        Map of column name and type for columns to consider in the inner dataframe
    unique_df (dict[str, Series]): 
        Map of binary/nominal columns and their unique values. This is stored 
        to avoid computing those unique values everytime as the inner dataset
        is likely not going to be modified

    Examples
    --------
    >>> from skmine.dssd import Table, ColumnType
    >>> import pandas
    >>> df = pandas.DataFrame({"A": 1.0, "D": [4, 3] * 2, "E": ["test", "train", "test", "train"]})
    >>> my_table = Table(df, {"A": ColumnType.NUMERIC, "D": ColumnType.NUMERIC, "E": ColumnType.NOMINAL})
    """

    def __init__(self, df: DataFrame, column_types: Dict[str, ColumnType]):
        self.df: DataFrame = df[column_types.keys()]
        self.column_types = column_types
        self.unique_df: dict[str, pandas.Series] = {col: df[col].dropna(inplace=False).unique() for col in column_types if (column_types[col] == ColumnType.NOMINAL or column_types[col] == ColumnType.BINARY)}
