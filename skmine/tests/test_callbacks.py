from ..callbacks import CallBacks, post

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