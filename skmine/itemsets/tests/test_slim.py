from ..slim import make_codetable
from ..slim import cover_one
from ..slim import generate_candidates
from ..slim import SLIM

from roaringbitmap import RoaringBitmap
import pandas as pd
import numpy as np

def test_make_cotetable():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    standard_codetable = make_codetable(D)
    pd.testing.assert_series_equal(
        standard_codetable.map(len),
        pd.Series([7, 7, 5], index=['A', 'B', 'C'])
    )


def test_cover_one():
    codetable_isets = list(map(
        frozenset,
        ['EBC', 'AFE', 'CD', 'AG', 'DF', 'E']
    ))
    cand = frozenset('ABCDEFG')
    cover = cover_one(codetable_isets, cand)
    assert cover == list(map(frozenset, ['EBC', 'AG', 'DF']))
    # TODO : output indices instead of elements


def test_generate_candidate_1():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    codetable = make_codetable(D)
    codetable.index = codetable.index.map(lambda e: frozenset([e]))
    new_candidates = generate_candidates(codetable)
    assert new_candidates == [frozenset('AB'), frozenset('BC')]

def test_generate_candidate_2():
    usage = list(map(RoaringBitmap, [
        range(7),
        [7],
        [8],
        range(6),
    ]))
    index = list(map(frozenset, ['AB', 'A', 'B', 'C']))
    codetable = pd.Series(usage, index=index)

    new_candidates = generate_candidates(codetable)
    assert new_candidates == [frozenset('ABC')]


def test_prefit():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    np.testing.assert_almost_equal(slim.model_size, 9.614, 3)
    np.testing.assert_almost_equal(slim.data_size, 29.798, 3)
