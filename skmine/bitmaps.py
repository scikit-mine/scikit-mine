"""
Bitmap definition for scikit-mine
"""
import platform

from roaringbitmap import RoaringBitmap as _RB

from sortedcontainers import SortedSet

class BitmapMock(SortedSet):  # pylint: disable=too-many-ancestors
    """
    Dummy implementation of a bitmap

    This inherits the python ``set`` class and provide some extra functions
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

if platform.system() == 'Windows':
    Bitmap = BitmapMock
else:
    Bitmap = _RB
