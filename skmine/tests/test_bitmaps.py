from ..bitmaps import _SortedSet

# test _SortedSet in any case


def test__SortedSet():
    bm = _SortedSet([1, 2, 2])
    bm2 = _SortedSet([1, 3])

    assert bm.intersection_len(bm2) == 1
    bm.flip_range(1, 5)
    assert repr(bm) == "[3, 4]"
