import numpy as np
import pandas as pd
import pytest
from pyroaring import BitMap as Bitmap
from sortedcontainers import SortedDict

from ..slim import SLIM, _to_vertical, _log2, cover


@pytest.fixture
def D():
    return pd.Series(["ABC"] * 5 + ["AB", "A", "B"])


@pytest.fixture
def codetable():
    return SortedDict({
        frozenset({'bananas'}): Bitmap([0, 1]),
        frozenset({'cookies'}): Bitmap([1, 2]),
        frozenset({'milk'}): Bitmap([0, 1]),
        frozenset({'butter'}): Bitmap([2]),
        frozenset({'tea'}): Bitmap([2])
    })


def to_tabular_df(D):
    return D.map(list).str.join("|").str.get_dummies(sep="|")


def _id(args):
    return args


def test_to_vertical(D):
    vert = _to_vertical(D)
    assert list(vert.keys()) == list("ABC")

    vert2 = _to_vertical(D, stop_items={"A"})
    assert list(vert2.keys()) == list("BC")


def test_cover():
    """
    A   B   C
    A   B
        B   C   D
    """
    # D corresponds to the items and the transactions in which they appear, it is the standard code table
    D = {
        "B": Bitmap([0, 1, 2]),
        "A": Bitmap([0, 1]),
        "C": Bitmap([0, 2]),
        "D": Bitmap([2])
    }
    # ct corresponds to the itemsets on which we want to calculate the cover
    ct = [
        frozenset("ABC"),
        frozenset("AB"),
        frozenset("BC"),
        frozenset("A"),
        frozenset("B"),
        frozenset("C"),
        frozenset("D")
    ]
    CTc = cover(D, ct)

    assert CTc[frozenset("ABC")] == Bitmap([0])
    assert CTc[frozenset("AB")] == Bitmap([1])  # AB appears only in tid_1 for usage because ABC is placed before in ct
    # so the AB of the first transaction has been covered by ABC
    assert CTc[frozenset("BC")] == Bitmap([2])
    assert CTc[frozenset("A")] == Bitmap()
    assert CTc[frozenset("B")] == Bitmap()
    assert CTc[frozenset("C")] == Bitmap()
    assert CTc[frozenset("D")] == Bitmap([2])


def test_log2():
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 0}
    d = pd.Series(data=d, index=['a', 'b', 'c', 'd'])
    logs = _log2(d)

    assert len(logs) == 4
    assert logs["a"] == 0
    assert logs["b"] == 1
    np.testing.assert_almost_equal(logs["c"], 1.58, 2)
    assert logs['d'] == 0  # attention : for a value of 0, this function returns 0


def test_update_no_candidate_and_usages(D):
    slim = SLIM().prefit(D)
    # Raises an AssertionError because the candidate and the uses cannot be None at the same time
    with pytest.raises(AssertionError):
        slim.update(candidate=None, model_size=10, data_size=20, usages=None) == AssertionError


def test_update_no_candidate(D):
    slim = SLIM().prefit(D)
    usages = {
        frozenset("ABC"): Bitmap([0, 1, 2, 3, 4]),
        frozenset("A"): Bitmap([5, 6]),
        frozenset("B"): Bitmap([5, 7])
    }
    slim.update(candidate=None, model_size=10, data_size=20, usages=usages)

    assert slim.model_size_ == 10
    assert slim.data_size_ == 20
    assert len(slim.codetable_) == 4  # C is always in the code table even if it is not in usages
    assert slim.codetable_.get(frozenset("ABC")) == Bitmap([0, 1, 2, 3, 4])
    assert slim.codetable_.get(frozenset("A")) == Bitmap([5, 6])
    assert slim.codetable_.get(frozenset("B")) == Bitmap([5, 7])
    assert slim.codetable_.get(frozenset("C")) == Bitmap([0, 1, 2, 3, 4])  # C has not been updated but is present


def test_update_no_usages(D):
    slim = SLIM().prefit(D)
    model_size = slim.model_size_
    data_size = slim.data_size_
    candidate = frozenset("ABC")
    slim.update(candidate=candidate, model_size=None, data_size=None, usages=None)

    assert slim.model_size_ != model_size
    assert slim.data_size_ != data_size
    assert len(slim.codetable_) == 4  # ABC has been added to the table in addition to the 3 singletons (A,B,C)
    assert slim.codetable_.get(frozenset("ABC")) == Bitmap([0, 1, 2, 3, 4])
    # the usages of A,B,C have been calculated
    assert slim.codetable_.get(frozenset("A")) == Bitmap([5, 6])
    assert slim.codetable_.get(frozenset("B")) == Bitmap([5, 7])
    assert slim.codetable_.get(frozenset("C")) == Bitmap()  # C has not been updated but is present


def test_update_to_drop(D):
    slim = SLIM().prefit(D)
    usages = {
        frozenset("AB"): Bitmap([0, 1, 2, 3, 4, 5]),
        frozenset("A"): Bitmap([6]),
        frozenset("B"): Bitmap([7]),
        frozenset("C"): Bitmap([0, 1, 2, 3, 4])
    }
    slim.update(candidate=None, model_size=15, data_size=20, usages=usages)
    assert slim.codetable_.get(frozenset("AB")) == Bitmap([0, 1, 2, 3, 4, 5])
    usages = {
        frozenset("ABC"): Bitmap([0, 1, 2, 3, 4]),
        frozenset("A"): Bitmap([5, 6]),
        frozenset("B"): Bitmap([7]),
        frozenset("C"): Bitmap()
    }
    slim.update(candidate=None, model_size=10, data_size=20, usages=usages)

    assert slim.model_size_ == 10
    assert slim.data_size_ == 20
    assert len(slim.codetable_) == 4
    assert slim.codetable_.get(frozenset("AB")) is None  # AB has been removed because it does not appear
    # in the new usages


def test_complex_evaluate():
    """
    A   B   C
    A   B
    A       C
        B
        B   C   D   E
    A   B   C   D   E
    """
    slim = SLIM()  # by default pruning=True
    D = ["ABC", "AB", "AC", "B", "BCDE", "ABCDE"]
    slim.prefit(D)

    u = {
        frozenset("ABC"): {0, 5},
        frozenset("AB"): {1},
        frozenset("BC"): {4},
        frozenset("DE"): {4, 5},
        frozenset("B"): {3},
        frozenset("A"): {2},
        frozenset("C"): {2},
        frozenset("D"): {},
        frozenset("E"): {},
    }

    u = {k: Bitmap(v) for k, v in u.items()}

    slim.codetable_.update(u)

    cand = frozenset("CDE")
    _, _, updated = slim.evaluate(cand)

    diff = {k: v for k, v in updated.items() if k in u and u[k] != v}

    # the BC itemset has been pruned from CTc because its use is now null, so in diff we have only "DE" and "B".
    assert len(diff) == 2
    assert len(updated[cand]) == 1  # {4}
    assert len(updated[frozenset("B")]) == 2  # {3} -> {3, 4}
    assert len(updated[frozenset("DE")]) == 1  # {4, 5} -> {5}


def test_complex_evaluate_2():
    """
    A   B   C
    A   B
    A       C
        B
        B   C   D   E
    A   B   C   D   E
    """
    slim = SLIM(pruning=False)
    D = ["ABC", "AB", "AC", "B", "BCDE", "ABCDE"]
    slim.prefit(D)

    u = {
        frozenset("CDE"): {4, 5},
        frozenset("AB"): {0, 1, 5},
        frozenset("BC"): {},
        frozenset("DE"): {},
        frozenset("B"): {3, 4},
        frozenset("A"): {2},
        frozenset("C"): {0, 2},
        frozenset("D"): {},
        frozenset("E"): {},
    }

    u = {k: Bitmap(v) for k, v in u.items()}

    slim.codetable_.update(u)

    cand = frozenset("ABC")
    _, _, updated = slim.evaluate(cand)

    diff = {k: v for k, v in updated.items() if k in u and u[k] != v}

    assert len(diff) == 4
    assert len(updated[cand]) == 2
    assert len(updated[frozenset("CDE")]) == 1  # {4, 5} -> {4}
    assert len(updated[frozenset("DE")]) == 1  # {} -> {5}
    assert len(updated[frozenset("AB")]) == 1  # {0, 1, 5} -> {1}
    assert len(updated[frozenset("C")]) == 1  # {0, 2} -> {2}


def test_standard_cover_order():
    D = ["ABC", "ABC", "ABCD", "C"]
    slim = SLIM().prefit(D)
    itemsets = slim.codetable_.keys()
    # support(A) = 3
    # support(B) = 3
    # support(C) = 4
    # support(D) = 1
    assert itemsets[0] == frozenset("C")  # C has the largest support
    assert itemsets[1] == frozenset("A")  # A has the same support as B but is before by lexicographic order
    assert itemsets[2] == frozenset("B")
    assert itemsets[3] == frozenset("D")  # D has the smallest support

    slim.codetable_.update({frozenset("AB"): Bitmap([0])})
    assert slim.codetable_.keys()[0] == frozenset("AB")  # AB is the longest itemset


def test_generate_candidate_1():
    D = ['ABC', 'AB', 'BCD']
    slim = SLIM().prefit(D)
    seen_cands = set(slim.codetable_.keys())
    candidates = slim.generate_candidates(stack=seen_cands)

    assert len(candidates) == 2  # only two candidates are returned out of the 6 potentials because only two have an
    # estimated gain > 0
    np.testing.assert_almost_equal(candidates[0][1], 2.39, 2)
    np.testing.assert_almost_equal(candidates[1][1], 2.39, 2)


@pytest.mark.parametrize("preproc", [to_tabular_df, _id])
def test_prefit(preproc):
    D = pd.Series(["ABC"] * 5 + ["BC", "B", "C"])
    D = preproc(D)
    slim = SLIM().prefit(D)
    np.testing.assert_almost_equal(slim.model_size_, 9.614, 3)
    np.testing.assert_almost_equal(slim.data_size_, 29.798, 3)
    assert len(slim.codetable_) == 3
    assert list(slim.codetable_) == list(map(frozenset, ["B", "C", "A"]))


def test_get_support(D):
    slim = SLIM().prefit(D)
    assert len(slim.get_support(*frozenset("ABC"))) == 5
    assert len(slim.get_support("C")) == 5
    assert slim.get_support.cache_info().currsize > 0


def test_compute_sizes_1(D):
    slim = SLIM()
    slim.prefit(D)
    CT = {
        frozenset("ABC"): Bitmap(range(0, 5)),
        frozenset("AB"): Bitmap([5]),
        frozenset("A"): Bitmap([6]),
        frozenset("B"): Bitmap([7]),
    }

    data_size, model_size = slim._compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.4, 2)
    np.testing.assert_almost_equal(model_size, 20.25, 2)


def test_compute_sizes_2(D):
    slim = SLIM()
    slim.prefit(D)
    CT = {
        frozenset("ABC"): Bitmap(range(0, 5)),
        frozenset("A"): Bitmap([5, 6]),
        frozenset("B"): Bitmap([5, 7]),
        frozenset("C"): Bitmap(),
    }

    data_size, model_size = slim._compute_sizes(CT)
    np.testing.assert_almost_equal(data_size, 12.92, 2)
    np.testing.assert_almost_equal(model_size, 12.876, 2)


@pytest.mark.parametrize("preproc,pass_y", ([to_tabular_df, False], [_id, True]))
def test_fit_pruning(D, preproc, pass_y):
    slim = SLIM(pruning=True)
    y = None if not pass_y else np.array([1] * len(D))
    D = preproc(D)
    self = slim.fit(D, y=y)
    assert list(self.codetable_) == list(map(frozenset, ["ABC", "A", "B", "C"]))


@pytest.mark.parametrize("preproc,pass_y", ([to_tabular_df, True], [_id, False]))
def test_fit_no_pruning(D, preproc, pass_y):
    slim = SLIM(pruning=False)
    y = None if not pass_y else np.array([1] * len(D))
    D = preproc(D)
    self = slim.fit(D, y=y)
    assert list(self.codetable_) == list(map(frozenset, ["ABC", "AC", "A", "B", "C"]))


def test_prune_usage_null(D):
    slim = SLIM(pruning=False).fit(D)
    # Codetable :
    # ABC : 0, 1, 2, 3, 4
    # AC : x
    # A : 5, 6
    # B : 5, 7
    # C : x

    new_codetable, new_data_size, new_model_size = slim._prune(slim.codetable_, slim.model_size_, slim.data_size_)

    # C is present because we do not prune itemsets of length 1 and AC is still removed because his usage is 0 and
    # because its length is 2
    assert list(new_codetable) == list(map(frozenset, ["ABC", "A", "B", "C"]))
    np.testing.assert_almost_equal(new_data_size, 12.92, 2)

    total_enc_size = new_data_size + new_model_size
    np.testing.assert_almost_equal(total_enc_size, 26, 0)


def test_force_prune_with_evaluate(D):
    slim = SLIM(pruning=True).prefit(D)
    _, _, usages = slim.evaluate(frozenset("AC"))
    slim.codetable_.update(usages)
    assert frozenset("AC") in usages
    _, _, usages = slim.evaluate(frozenset("ABC"))
    assert frozenset("ABC") in usages
    # AC has been removed because after the addition of ABC, its usage became null.
    assert frozenset("AC") not in usages


def test_prune(D):
    slim = SLIM(pruning=False).prefit(D)
    # Add AB in the codetable -> usage(AB) = 6
    _, _, usages = slim.evaluate(frozenset("AB"))
    slim.codetable_.update(usages)

    # Add ABC in the codetable -> usage(AB) = 1
    data_size, model_size, usages = slim.evaluate(frozenset("ABC"))

    # AB no longer improves compression, pruning must remove it
    new_codetable, new_data_size, new_model_size = slim._prune(
        usages, slim.model_size_, slim.data_size_
    )

    assert data_size + model_size > new_data_size + new_model_size
    assert list(new_codetable) == list(map(frozenset, ["ABC", "A", "B", "C"]))


def test_get_code_length(D):
    slim = SLIM(pruning=True).fit(D)

    new_D = pd.Series(["AB"] * 2 + ["ABD", "AC", "B"])
    new_D = new_D.str.join("|").str.get_dummies(sep="|")

    code_l = slim.get_code_length(new_D)
    print(code_l.values)
    assert code_l.dtype == np.float32
    assert len(code_l) == len(new_D)
    np.testing.assert_array_almost_equal(code_l.values, np.array([4.23, 4.23, 4.23, 5.81, 2.11]), decimal=2)


def test_decision_function(D):
    slim = SLIM(pruning=True).fit(D)

    new_D = pd.Series(["AB"] * 2 + ["ABD", "AC", "B"])
    new_D = new_D.str.join("|").str.get_dummies(sep="|")

    prob = slim.decision_function(new_D)
    print(prob.values)
    assert prob.dtype == np.float32
    assert len(prob) == len(new_D)
    np.testing.assert_array_almost_equal(prob.values, np.exp(-0.2 * np.array([4.23, 4.23, 4.23, 5.81, 2.11])),
                                         decimal=2)


def test_reconstruct(D):
    slim = SLIM().fit(D)
    s = slim.reconstruct().map("".join)  # originally a string so we have to join
    true_s = pd.Series(["ABC"] * 5 + ["AB", "A", "B"])
    pd.testing.assert_series_equal(s, true_s)
