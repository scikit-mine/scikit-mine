from ..bitmaps import BitmapMock

# test BitmapMock in any case


def test_bitmapmock():
    bm = BitmapMock([1, 2, 2])
    bm2 = BitmapMock([1, 3])

    assert bm.intersection_len(bm2) == 1
    bm.flip_range(1, 5)
    assert repr(bm) == "[3, 4]"

    bm3 = BitmapMock([1, 6])
    assert list(~bm3) == [2, 3, 4, 5]
