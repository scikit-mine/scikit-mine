"""
Bitmap definition for scikit-mine
"""
import platform

from sortedcontainers import SortedSet


class BitmapMock(SortedSet):  # pylint: disable=too-many-ancestors
    """
    Dummy implementation of a bitmap

    This inherits the ``SortedSet`` class and provide some extra functions
    to ensure compatibility with other -performant- bitmap implementations
    """

    def intersection_len(self, other):
        """Returns length of the intersection"""
        return len(self & other)

    def flip_range(self, start, stop):
        """In-place negation for range(start, stop)"""
        for e in range(start, stop):
            if e in self:
                self.discard(e)
            else:
                self.add(e)

    def __repr__(self):
        return "[{}]".format(", ".join(map(str, self)))

    def __invert__(self):
        return type(self)(set(range(self[0], self[-1])) - self)

    max = lambda self: self[-1]
    min = lambda self: self[0]

    __str__ = __repr__


if platform.system() == "Windows":
    Bitmap = BitmapMock
else:
    from roaringbitmap import RoaringBitmap as Bitmap  # pylint: disable= unused-import
