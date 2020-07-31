from ..callbacks import CallBacks, post, _print_candidates_size, _print_positive_gain
from ..callbacks import has_self_assigment

from io import StringIO
import sys

import pytest

class Obj():
    def __init__(self, x):
        self.x = x

    def method_a(self, e):
        self.x = e
        return e + 1

    def method_b(self, e):
        return e, e + self.x

@pytest.fixture
def obj():
    return Obj(x=3)


def test_post_1(obj):
    stack = []
    def callback(self, res):
        _, res = res
        stack.append(res)

    setattr(obj, 'method_b', post(obj, 'method_b', callback))

    obj.method_a(4)  # set self.x to 4
    obj.method_b(10)   # append res to stack

    assert stack == [14]


def test_post_2(obj):
    stack = []
    dec_func = post(obj, 'method_b', stack.extend)
    setattr(obj, 'method_b', dec_func)
    assert obj.method_b(2) == (2, 5)
    assert stack == [2, 5] 


def test_callbacks(obj):
    stack1, stack2 = list(), list()
    callbacks = CallBacks(method_a=stack1.append, method_b=stack2.extend)

    obj2 = callbacks(obj)
    assert obj2 is obj

    obj.method_a(10)
    obj.method_a(2)
    obj.method_b(2)

    assert stack1 == [11, 3]
    assert stack2 == [2, 4]


def test_callbacks_2(obj):
    stack = list()
    def cb(self):
        stack.append(self.x)

    callbacks = CallBacks(method_a=cb)
    callbacks(obj)

    assert obj.method_a(10) == 11
    assert stack == [10]


def test_test():
    class A():
        def f(self):
            return 10
    stack = list()
    callbacks = CallBacks(f=stack.append)
    a = A()
    _ = callbacks(a)
    assert a.f() == 10
    assert stack == [10]

def test_multiple_return():
    class A():
        def f(self, a):
            return (2, 10, 3)

    stack = list()
    def cb(self, a, b, *_):
        stack.append(b)

    callbacks = CallBacks(f=cb)
    a = A()
    callbacks(a)
    assert a.f(None) == (2, 10, 3)
    assert stack == [10]


def test_lambda(obj):
    stack = list()
    callbacks = CallBacks(method_a=lambda res: stack.append(res))
    callbacks(obj)
    assert obj.method_a(10) == 11
    assert obj.x == 10
    assert stack == [11]

def test_multiargs(obj):
    stack = list()
    def cb(data_size, model_size, *args):
        stack.append((data_size, model_size))

    callbacks = CallBacks(method_b=cb)
    callbacks(obj)
    assert obj.method_b(10) == (10, 13)
    assert stack == [(10, 13)]

def test_wrong_args(obj):
    with pytest.raises(TypeError):
        CallBacks({2: lambda: 3})

    with pytest.raises(TypeError):
        CallBacks(f=3)

    callbacks = CallBacks(method_c=lambda e: 3)
    with pytest.raises(ValueError):
        callbacks(obj)    # no method_c for obj


class MDLObj():
    verbose = True
    data_size_ = 3
    model_size_ = 4

def test_mdl_cand_size():
    o = StringIO()
    sys.stdout = o
    obj = MDLObj()
    _print_candidates_size(obj, [2])
    sys.stdout = sys.__stdout__
    assert '1' in o.getvalue()

def test_mdl_cand_size():
    o = StringIO()
    sys.stdout = o
    obj = MDLObj()
    _print_positive_gain(obj, 1, 2)
    sys.stdout = sys.__stdout__
    assert 'data' in o.getvalue()
    assert 'model' in o.getvalue()


def test_check_no_self():
    def f(self, a):
        self.a_b = a
        return a

    assert has_self_assigment(f)

    def f(self, a):
        return a
    assert not has_self_assigment(f)

    def f(self, a):
        self = None
        return a

    assert has_self_assigment(f)

    with pytest.raises(ValueError):
        CallBacks(f=f)
