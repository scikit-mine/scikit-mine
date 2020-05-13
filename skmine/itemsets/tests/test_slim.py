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

def test_generate_candidate_stack():
    usage = list(map(RoaringBitmap, [
        range(6),
        [6, 7],
        [6, 8],
        [],
    ]))
    index = list(map(frozenset, ['ABC', 'A', 'B', 'C']))
    codetable = pd.Series(usage, index=index)

    new_candidates = generate_candidates(codetable, stack={frozenset('AB')})
    assert new_candidates == []


def test_cover_order_pos_1():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    codetable = ['A', 'B', 'C']
    codetable = list(map(frozenset, codetable))
    cand = frozenset('ABC')

    pos = slim._get_cover_order_pos(codetable, cand)

    assert pos == 0
    # empty dict because checking supports was not necessary
    assert not slim.supports


def test_cover_order_pos_2():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    codetable = ['ABC', 'B', 'C']
    codetable = list(map(frozenset, codetable))
    cand = frozenset('AB')

    pos = slim._get_cover_order_pos(codetable, cand)

    assert pos == 1
    assert cand in slim.supports.keys()


def test_prefit():
    D = ['ABC'] * 5 + ['BC', 'B', 'C']
    slim = SLIM()
    slim._prefit(D)
    np.testing.assert_almost_equal(slim.model_size, 9.614, 3)
    np.testing.assert_almost_equal(slim.data_size, 29.798, 3)
    assert len(slim.codetable) == 3
    assert slim.codetable.dtype == np.object
    assert slim.codetable.index.tolist() == list(map(frozenset, ['B', 'C', 'A']))

def test_get_standard_size_1():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    CT_index = ['ABC', 'AB', 'A', 'B']
    codes = slim.get_standard_codes(CT_index)
    pd.testing.assert_series_equal(
        codes,
        pd.Series([4.32, 4.32, 1.93], index=list('ABC')),
        check_less_precise=2
    )

def test_get_standard_size_2():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    CT_index = ['ABC', 'A', 'B']
    codes = slim.get_standard_codes(CT_index)
    pd.testing.assert_series_equal(
        codes,
        pd.Series([2.88, 2.88, 1.93], index=list('ABC')),
        check_less_precise=2
    )


def test_compute_sizes_1():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    CT = pd.Series({
        frozenset('ABC'): RoaringBitmap(range(0, 5)),
        frozenset('AB'): RoaringBitmap([5]),
        frozenset('A'): RoaringBitmap([6]),
        frozenset('B'): RoaringBitmap([7]), 
    })

    data_size, model_size = slim.compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.4, 2)
    np.testing.assert_almost_equal(model_size, 20.25, 2)


def test_compute_sizes_2():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    slim = SLIM()
    slim._prefit(D)
    CT = pd.Series({
        frozenset('ABC'): RoaringBitmap(range(0, 5)),
        frozenset('A'): RoaringBitmap([5, 6]),
        frozenset('B'): RoaringBitmap([5, 7]),
    })

    data_size, model_size = slim.compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.92, 2)
    np.testing.assert_almost_equal(model_size, 12.876, 2)

def test_fit_no_pruning():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    D = pd.Series(D)
    slim = SLIM(pruning=False)
    self = slim.fit(D)
    assert self.codetable.index.tolist() == list(map(frozenset, ['ABC', 'AB', 'A', 'B', 'C']))

def test_fit():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    D = pd.Series(D)
    slim = SLIM(pruning=True)
    self = slim.fit(D)
    assert self.codetable.index.tolist() == list(map(frozenset, ['ABC', 'A', 'B', 'C']))

def test_prune():
    D = ['ABC'] * 5 + ['AB', 'A', 'B']
    D = pd.Series(D)

    slim = SLIM(pruning=False).fit(D)
    prune_set = slim.codetable.loc[[frozenset('AB')]]

    
    new_codetable, new_data_size, new_model_size = slim._prune(
        slim.codetable, D, prune_set, slim.model_size, slim.data_size
    )

    assert new_codetable.index.tolist() == list(map(frozenset, ['ABC', 'A', 'B', 'C']))
    np.testing.assert_almost_equal(new_data_size, 12.92, 2)

    total_enc_size = new_data_size + new_model_size
    np.testing.assert_almost_equal(total_enc_size, 26, 0)