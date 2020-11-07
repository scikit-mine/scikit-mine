"""
Bitmap definition for scikit-mine
"""
import platform

from sortedcontainers import SortedSet
from pyroaring import BitMap as _RB


class _SortedSet(SortedSet):  # pylint: disable=too-many-ancestors
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

    __str__ = __repr__


class _BitMap(_RB):
    def intersection_len(self, other):
        return len(self & other)


if platform.system() == "Windows":
    Bitmap = _SortedSet
else:
    Bitmap = _BitMap
