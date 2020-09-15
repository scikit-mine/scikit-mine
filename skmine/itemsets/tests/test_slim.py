from ..slim import cover
from ..slim import generate_candidates
from ..slim import SLIM
from ...preprocessing.transaction_encoder import TransactionEncoder
from ...bitmaps import Bitmap

import pytest
import pandas as pd
import numpy as np
from sortedcontainers import SortedDict


def dense_D():
    D = ["ABC"] * 5 + ["AB", "A", "B"]
    D = TransactionEncoder().fit_transform(D)
    return D


def sparse_D():
    D = ["ABC"] * 5 + ["AB", "A", "B"]
    D = TransactionEncoder(sparse_output=True).fit_transform(D)
    return D


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_cover_1(D):
    isets = [frozenset(_) for _ in ("ABC", "A", "B", "C")]

    covers = cover(isets, D)
    pd.testing.assert_series_equal(
        pd.Series([5, 2, 2, 0], index=isets),
        covers.map(len),
    )


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_cover_2(D):
    isets = [frozenset(_) for _ in ("ABC", "AB", "A", "B", "C")]

    covers = cover(isets, D)
    pd.testing.assert_series_equal(
        pd.Series([5, 1, 1, 1, 0], index=isets),
        covers.map(len),
    )


def test_cover_3():
    D = ["ABC"] * 5 + ["AB", "A", "B", "DE", "CDE"]
    D = TransactionEncoder().fit_transform(D)

    isets = [frozenset(_) for _ in ("ABC", "DE", "A", "B", "C")]

    covers = cover(isets, D)
    pd.testing.assert_series_equal(
        pd.Series([5, 2, 2, 2, 1], index=isets),
        covers.map(len),
    )


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_cover_4(D):
    isets = [frozenset(_) for _ in ("BC", "AB", "A", "B", "C")]

    covers = cover(isets, D)
    pd.testing.assert_series_equal(
        pd.Series([5, 1, 6, 1, 0], index=isets),
        covers.map(len),
    )


def test_complex_evaluate():
    """
    A   B   C
    A   B
    A       C
        B
        B   C   D   E
    A   B   C   D   E
    """
    slim = SLIM()
    D = ["ABC", "AB", "AC", "B", "BCDE", "ABCDE"]
    slim._prefit(TransactionEncoder().fit_transform(D))

    u = {
        frozenset("ABC"): {0, 5},
        frozenset("AB"): {1},
        frozenset("BC"): {4},
        frozenset("DE"): {4, 5},
        frozenset("A"): {2},
        frozenset("B"): {3},
        frozenset("C"): {2},
        frozenset("D"): {},
        frozenset("E"): {},
    }

    u = {k: Bitmap(v) for k, v in u.items()}

    slim.codetable_.update(u)

    cand = frozenset("CDE")
    data_size, model_size, updated, decreased = slim.evaluate(cand)

    assert decreased == [frozenset("BC"), frozenset("DE")]

    assert len(updated) == 4
    assert len(updated[cand]) == 1  # {4}
    assert len(updated[frozenset("BC")]) == 0  # {4} -> {}
    assert len(updated[frozenset("B")]) == 2  # {3} -> {3, 4}
    assert len(updated[frozenset("DE")]) == 1  # {4, 5} -> {5}


def test_generate_candidate_1():
    D = ["ABC"] * 5 + ["AB", "A", "B"]

    codetable = SortedDict(
        {
            frozenset("A"): Bitmap(range(0, 7)),
            frozenset("B"): Bitmap([0, 1, 2, 3, 4, 5, 7]),
            frozenset("C"): Bitmap(range(0, 5)),
        }
    )

    new_candidates = generate_candidates(codetable)
    assert new_candidates == [
        (frozenset("AB"), 6),
        (frozenset("BC"), 5),
    ]


def test_generate_candidate_2():
    usage = [Bitmap(_) for _ in (range(6), [6], [7], range(5))]

    index = list(map(frozenset, ["AB", "A", "B", "C"]))
    codetable = SortedDict(zip(index, usage))

    new_candidates = generate_candidates(codetable)
    assert new_candidates == [(frozenset("ABC"), 5)]


def test_generate_candidate_stack():
    usage = [Bitmap(_) for _ in (range(6), [6, 7], [6, 8], [])]

    index = list(map(frozenset, ["ABC", "A", "B", "C"]))

    codetable = SortedDict(zip(index, usage))

    new_candidates = generate_candidates(codetable, stack={frozenset("AB")})
    assert new_candidates == []


def test_cover_order_pos():
    """ lower size but higher support"""
    supports = {
        "ABC": 5,
        "AB": 7,
        "BC": 8,
        "B": 10,
    }

    codetable = SortedDict(
        lambda e: (-len(e), -supports[e], e),
        {
            "ABC": 2,
            "BC": 11,
            "B": 10,
        },  # Note: values have no imporante here
    )
    cand = "AB"

    pos = codetable.bisect(cand)

    assert pos == 2


def test_prefit():
    D = ["ABC"] * 5 + ["BC", "B", "C"]
    D = TransactionEncoder().fit_transform(D)
    slim = SLIM()
    slim._prefit(D)
    np.testing.assert_almost_equal(slim.model_size_, 9.614, 3)
    np.testing.assert_almost_equal(slim.data_size_, 29.798, 3)
    assert len(slim.codetable) == 3
    assert slim.codetable.index.tolist() == list(map(frozenset, ["B", "C", "A"]))


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_get_standard_size_1(D):
    slim = SLIM()
    slim._prefit(D)
    CT_index = ["ABC", "AB", "A", "B"]
    codes = slim.get_standard_codes(CT_index)
    pd.testing.assert_series_equal(
        codes, pd.Series([4.32, 4.32, 1.93], index=list("ABC")), check_less_precise=2
    )


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_get_standard_size_2(D):
    slim = SLIM()
    slim._prefit(D)
    CT_index = ["ABC", "A", "B"]
    codes = slim.get_standard_codes(CT_index)
    pd.testing.assert_series_equal(
        codes, pd.Series([2.88, 2.88, 1.93], index=list("ABC")), check_less_precise=2
    )


@pytest.mark.parametrize("D", [dense_D()])
def test_compute_sizes_1(D):
    slim = SLIM()
    slim._prefit(D)
    CT = {
        frozenset("ABC"): Bitmap(range(0, 5)),
        frozenset("AB"): Bitmap([5]),
        frozenset("A"): Bitmap([6]),
        frozenset("B"): Bitmap([7]),
    }

    data_size, model_size = slim.compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.4, 2)
    np.testing.assert_almost_equal(model_size, 20.25, 2)


@pytest.mark.parametrize("D", [dense_D()])
def test_compute_sizes_2(D):
    slim = SLIM()
    slim._prefit(D)
    CT = {
        frozenset("ABC"): Bitmap(range(0, 5)),
        frozenset("A"): Bitmap([5, 6]),
        frozenset("B"): Bitmap([5, 7]),
        frozenset("C"): Bitmap(),
    }

    data_size, model_size = slim.compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.92, 2)
    np.testing.assert_almost_equal(model_size, 12.876, 2)


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_fit_no_pruning(D):
    slim = SLIM(pruning=False)
    self = slim.fit(D)
    assert list(self.codetable_) == list(map(frozenset, ["ABC", "AB", "A", "B", "C"]))


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_fit(D):
    slim = SLIM(pruning=True)
    self = slim.fit(D)
    assert list(self.codetable_) == list(map(frozenset, ["ABC", "A", "B", "C"]))


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_fit_ndarray(D):
    slim = SLIM(pruning=True)
    self = slim.fit(D.values)
    assert list(self.codetable_) == list(map(frozenset, [[0, 1, 2], [0], [1], [2]]))


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_prune(D):
    slim = SLIM(pruning=False).fit(D)
    prune_set = [frozenset("AB")]

    new_codetable, new_data_size, new_model_size = slim._prune(
        slim.codetable_, D, prune_set, slim.model_size_, slim.data_size_
    )

    assert list(new_codetable) == list(map(frozenset, ["ABC", "A", "B", "C"]))
    np.testing.assert_almost_equal(new_data_size, 12.92, 2)

    total_enc_size = new_data_size + new_model_size
    np.testing.assert_almost_equal(total_enc_size, 26, 0)


@pytest.mark.parametrize("D", [dense_D(), sparse_D()])
def test_prune_empty(D):
    slim = SLIM(pruning=False).fit(D)
    prune_set = [frozenset("ABC")]

    # nothing to prune so we should get the exact same codetable

    new_codetable, new_data_size, new_model_size = slim._prune(
        slim.codetable_, D, prune_set, slim.model_size_, slim.data_size_
    )

    assert list(new_codetable) == list(map(frozenset, ["ABC", "AB", "A", "B", "C"]))


def test_decision_function():
    te = TransactionEncoder(sparse_output=False)
    D = te.fit_transform(["ABC"] * 5 + ["AB", "A", "B"])
    slim = SLIM(pruning=True).fit(D)

    new_D = ["AB"] * 2 + ["ABD", "AC", "B"]
    new_D = te.fit_transform(new_D)

    dists = slim.decision_function(new_D)
    assert dists.dtype == np.float32
    assert len(dists) == len(new_D)
    np.testing.assert_array_almost_equal(
        dists.values, np.array([-1.17, -1.17, -1.17, -2.17, -2.17]), decimal=2
    )


def test_fit_sklearn():
    D = dense_D()
    y = np.array([1] * len(D))
    slim = SLIM().fit(D, y)
    assert slim.standard_codetable_.index.tolist() == ["A", "B", "C"]

    slim = SLIM().fit(D.values, y)
