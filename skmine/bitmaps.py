"""
Bitmap definition for scikit-mine
"""
import platform

from roaringbitmap import RoaringBitmap as _RB

class BitmapMock(set):
    """
    Dummy implementation of a bitmap

    This inherits the python ``set`` class and provide some extra functions
    to ensure compatibility with other -performant- bitmap implementations
    """
    def intersection_len(self, other):
        """Returns length of the intersection"""
        return len(self & other)

if platform.system() == 'Windows':
    Bitmap = BitmapMock
else:
    Bitmap = _RB
