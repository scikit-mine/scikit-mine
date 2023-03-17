import pandas as pd
import pytest

from ..slim_classifier import SlimClassifier


@pytest.fixture
def data():
    y = pd.Series([0, 0, 1, 2, 3, 2])
    X = pd.Series([[5, 8, 9], [5, 8, 7], [4, 7, 10], [4, 8, 10], [4, 7, 9], [4, 7, 10]])
    return X, y


def test_template_classifier(data):
    y = data[1]
    X = data[0]
    items = set(i for t in X for i in t)
    clf = SlimClassifier(items=items)

    assert clf.items == {4, 5, 7, 8, 9, 10}
    assert clf.pruning is False

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'classes_X_')
    assert hasattr(clf, 'models_')

    y_pred = clf.predict(X)
    print(X.shape)
    assert y_pred.shape == (X.shape[0],)


def test_models_(data):
    y = data[1]
    X = data[0]
    items = set(i for t in X for i in t)
    clf = SlimClassifier(items=items)
    clf.fit(X, y)

    assert len(clf.models_) == 4


def test_classes_(data):
    y = data[1]
    X = data[0]
    items = set(i for t in X for i in t)
    clf = SlimClassifier(items=items)
    clf.fit(X, y)

    assert len(clf.classes_) == 4
    assert [0, 1, 2, 3] in clf.classes_


def test_classes_X_(data):
    y = data[1]
    X = data[0]
    items = set(i for t in X for i in t)
    clf = SlimClassifier(items=items)
    clf.fit(X, y)

    assert len(clf.classes_X_) == 4
    assert clf.classes_X_[0] == [X[0], X[1]]
    assert clf.classes_X_[1] == [X[2]]
    assert clf.classes_X_[2] == [X[3], X[5]]
    assert clf.classes_X_[3] == [X[4]]
