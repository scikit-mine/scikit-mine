from ..sequences import GoKrimp, to_vertical, compress_size, best_compressing
import pytest
import pandas as pd


@pytest.fixture
def D():
    return pd.Series(["acabcb", "cabcab"])


def test_to_vertical(D):
    vert, seq_lengths = to_vertical(D)
    assert isinstance(vert, dict)
    assert len(vert) == 3  # alphabet of size 3

    start, end = seq_lengths[:2]  # second transaction
    assert max(vert["a"].clamp(3, 100)) == 10  # last a at position 10


@pytest.mark.parametrize(
    "pattern,max_gap,size",
    [
        ("ab", 3, 10),
        ("ab", 1, 11),
        ("cab", 2, 9),
    ],
)
def test_compress_size(D, pattern, max_gap, size):
    vert, seq_lengths = to_vertical(D)
    D_size, updated_d = compress_size(vert, pattern, seq_lengths, max_gap=max_gap)
    assert D_size == size
    assert tuple(pattern) in updated_d
    for e in pattern:
        assert e in updated_d


def test_compress_empty_bitmaps(D):
    pass  # TODO


def test_best_compressing(D):
    """
    cab has a cost of 9
    ab has a cost of 10
    cb has a cost of 10
    """
    vert, seq_lengths = to_vertical(D)
    best_pattern, update_d = best_compressing(vert, seq_lengths, max_gap=10)
    assert best_pattern == tuple("cab")
    assert len(update_d["a"]) == 1  # all but one 'a' has been consumed by 'cab'
    assert update_d["a"].max() == 0  # last a is a pos one
    assert update_d["c"].max() == 4  # last c is a pos one


@pytest.mark.parametrize("k", [10, 2])
def test_fit(D, k):
    gk = GoKrimp(k=k)
    gk.fit(D)
    codetable = gk.discover()
    assert len(codetable) <= k
    assert codetable.index.map(len).is_monotonic_decreasing


def test_reconstruct(D):
    gk = GoKrimp().fit(D)
    r = gk.reconstruct().str.join("")
    pd.testing.assert_series_equal(D, r)
