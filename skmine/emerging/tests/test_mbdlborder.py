import pandas as pd

from ..mbdldorber import MBDLLBorder
from ..mbdldorber import border_diff
from ..mbdldorber import borders_to_patterns
from ..mbdldorber import mbdllborder


def test_borders_diff():
    U = {1, 2, 3, 4}
    S = [{3, 4}, {2, 4}, {2, 3}]

    L, U = border_diff(U, S)
    assert L == [{1}, {2, 3, 4}]
    assert U == {1, 2, 3, 4}


def test_mbdllborder():
    isets1 = [{2, 3, 5}, {3, 4, 6, 7, 8}, {2, 4, 5, 8, 9}]
    isets2 = [{1, 2, 3, 4}, {6, 7, 8}]

    borders = mbdllborder(isets1, isets2)

    left_border, right_border = borders[0]
    assert left_border == [{1}, {2, 3, 4}]
    assert right_border == {1, 2, 3, 4}


def test_borders_to_patterns():
    left_border = [{1}, {2, 3, 4}]
    right_border = {1, 2, 3, 4}

    patterns = borders_to_patterns(left_border, right_border)

    patterns = list(patterns)

    assert patterns == [
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (1, 2),
        (1, 3),
        (1, 4),
    ]


def test_border_to_patterns_min_size():
    left_border = [{1}, {2, 3, 4}]
    right_border = {1, 2, 3, 4}

    patterns = borders_to_patterns(left_border, right_border, min_size=3)

    patterns = list(patterns)

    assert patterns == [
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
    ]


def test_discover():
    D = pd.Series(
        [
            ["banana", "chocolate"],
            ["sirup", "tea"],
            ["chocolate", "banana"],
            ["chocolate", "milk", "banana"],
        ]
    )

    y = pd.Series(
        [
            "food",
            "drink",
            "food",
            "drink",
        ],
        dtype="category",
    )

    ep = MBDLLBorder(min_growth_rate=1.2)
    patterns = ep.fit_discover(D, y, min_size=2)

    assert isinstance(patterns, pd.Series)
