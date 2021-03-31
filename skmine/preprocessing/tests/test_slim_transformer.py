import numpy as np
import pytest

from .. import SLIMTransformer


@pytest.mark.parametrize("strategy", ("one-hot", "codes"))
def test_fit_transform(strategy):
    D = [["banana", "cookies"], ["banana", "milk"], ["banana", "milk", "cookies"]]

    with pytest.raises(ValueError):
        SLIMTransformer().fit(D)  # default k too big

    st = SLIMTransformer(k=2, strategy=strategy)
    new_D = D + [["cookies", "milk", "apple"]]  # apple unseen
    mat = st.fit(D).transform(new_D)
    assert not "apple" in mat.columns
    assert mat.shape == (4, 2)

    uniques = np.unique(mat)
    if strategy == "one-hot":
        np.testing.assert_array_equal(uniques, [0, 1])


def test_wrong_strategy():
    with pytest.raises(ValueError):
        SLIMTransformer(strategy="random")
