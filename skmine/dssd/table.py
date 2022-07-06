from typing import Dict
from pandas import DataFrame
import pandas

from .custom_types import ColumnType

class Table:
    """A wrapper around pandas dataframe to hold some usefull attributes often used by the dssd algorithm ['s parts]

    Attributes:
    -----------
        df (DataFrame): inner pandas dataframe
        column_types (dict[str, ColumnType]): map of column name and type for columns to consider in the inner dataframe
        unique_df (dict[str, Series]): map of binary/nominal columns and their unique values. this is stored so as to avoid computing those unique values everytime they are needed
    """

    def __init__(self, df: DataFrame, column_types: Dict[str, ColumnType]):
        self.df: DataFrame = df[column_types.keys()]
        self.column_types = column_types
        self.unique_df: dict[str, pandas.Series] = {col: df[col].dropna(inplace=False).unique() for col in column_types if (column_types[col] == ColumnType.NOMINAL or column_types[col] == ColumnType.BINARY)}


    def __len__(self):
        return len(self.df)

