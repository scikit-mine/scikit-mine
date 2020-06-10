from ..mdlp_discretizer import (
    MDLPVectDiscretizer,
    MDLPDiscretizer,
    generate_cut_point,
)
import numpy as np

def test_generate_cut_points():
    # here y follow a linear function
    # so the cut_point is half the length of y
    y = np.arange(10) * 3
    cut_point = generate_cut_point(y, 2, 8)
    assert cut_point == 5  # linear function so


def test_vect_discretizer():
    X = np.array([5.1, 5.4, 5.4, 4.8, 5., 7., 5., 5.9, 5.5, 5.5, 6.3, 6.5, 6.9])
    y = np.array([0]* 5 + [1] * 5 + [2] * 5)

    disc = MDLPVectDiscretizer()
    disc.fit(X, y)
    np.testing.assert_array_almost_equal(disc.cut_points_, np.array([5.5]), decimal=1)


def test_mdlp_discretizer():
    X = np.ones((100, 4))
    X *= np.arange(100).reshape((100, 1))
    y = np.array([0] * 25 + [1] * 25 + [2] * 50)

    disc = MDLPDiscretizer()
    disc.fit(X, y)

    assert len(disc.cut_points_) == 4
    assert len(disc.discretizers_) == 4

    assert len(disc.discretizers_[0].cut_points_) == 2  # 3 parts --> 2 cut points
