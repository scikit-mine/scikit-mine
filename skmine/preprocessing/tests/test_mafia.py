from ..mafia import Mafia
import pytest


@pytest.fixture
def D():
    return [
        (1, 2, 3, 4),
        (2, 3),
        (3, 4),
        (2, 3),
        (1, 2, 3, 4),
        (1, 2),
        (2, 3),
    ]

def test_prefit(D):
    mafia = Mafia(min_supp=3)
    mfi = mafia._prefit(D)

    assert mfi.item_to_tids_ is not None
    assert list(mfi.item_to_tids_.keys()) == [1, 2, 3, 4]


def test_fit(D):
    mafia = Mafia(min_supp=3)
    mfi = mafia.fit_discover(D)

    assert mfi.map(len).to_dict() == {
        (1, 2) : 3,
        (2, 3) : 5,
        (3, 4) : 3,
    }
