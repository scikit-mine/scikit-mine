from ..mbdldorber import MBDLLBorder
from ..mbdldorber import border_diff
from ..mbdldorber import mbdllborder

import pandas as pd

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

    left_borders, right_borders = zip(*borders)
    assert left_borders == ([{1}, {2, 3, 4}],)
    assert right_borders == ({1, 2, 3, 4},)


def test_fit():
    D = pd.Series([
        ['banana', 'chocolate'],
        ['sirup', 'tea'],
        ['chocolate', 'bread'],
        ['chocolate', 'milk'],
        ['sirup', 'milk'],
        ['milk', 'sirup', 'tea']
    ])

    y = pd.Series([
        'food',
        'drink',
        'food',
        'drink',
        'drink',
        'drink'
    ], dtype='category')

    ep = MBDLLBorder(min_growth_rate=1.2)
    ep.fit(D, y)

    # TODO : 
    _, discriminative_patterns = zip(*ep.borders_)
    assert discriminative_patterns == (
        {'chocolate', 'banana'},
        {'chocolate', 'bread'},
    )


def test_discover():
    D = pd.Series([
        ['banana', 'chocolate'],
        ['sirup', 'tea'],
        ['chocolate', 'bread'],
        ['chocolate', 'milk'],
        ['sirup', 'milk'],
        ['milk', 'sirup', 'tea'],
    ])
    
    y = pd.Series([
        'food',
        'drink',
        'food',
        'drink',
        'drink',
        'drink'
    ], dtype='category')

    ep = MBDLLBorder(min_growth_rate=1.2)
    patterns = ep.fit_discover(D, y)

    pd.testing.assert_series_equal(
        patterns,
        pd.Series([
            {'chocolate', 'banana'},
            {'chocolate', 'bread'},
        ])
    )