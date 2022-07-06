import pandas as pd
from ..custom_types import ColumnType
from ..table import Table


def test_table_creation():
    df = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": [1.0] * 4,
        "D": [4, 3] * 2,
        "E": ["test", "train", "test", "train"],
        "F": "foo",
    })

    my_table = Table(df, {"A": ColumnType.NUMERIC, "C": ColumnType.NUMERIC, "D": ColumnType.NUMERIC, "E": ColumnType.NOMINAL, "F": ColumnType.NOMINAL})
    
    # B is not part of the column types given during table creation so that column was simply discarded
    assert "B" not in my_table.df.columns

