"""
Callback API for scikit-mine
"""
from inspect import signature, getsource

import re


def has_self_assigment(f):
    """
    Parameters
    ----------
    f: callable

    check if no assignement is made on the ``self`` keyword
    """
    try:
        p = r"self(\.\w+)?\s*=.*"
        return any(re.finditer(p, getsource(f)))
    except TypeError:
        return False


def _get_params(fn):
    assert callable(fn)
    try:
        sig = signature(fn)
        params = list(sig.parameters)
    except ValueError:
        params = list()
    return params


def post(self, func_name, callback):
    """decorator, call callback on returned values, after main function"""
    func = getattr(self, func_name)
    assert callable(func)
    callback_params = _get_params(callback)

    def _(*args, **kwargs):
        res = func(*args, **kwargs)
        if "self" in callback_params:
            _w = (res,) if len(callback_params) == 2 else res
            args_ = (self,) + (_w if len(callback_params) > 1 else tuple())
        else:
            args_ = res
        try:
            callback(*args_)
        except TypeError:
            callback(args_)  # list.extend
        return res

    return _


class CallBacks(dict):
    """
    A collection of callbacks

    Works by defining functions to be called after the execution of the function they target

    Parameters
    ----------
    key-value pairs
        Keys must be string, values must be callables

    Examples
    --------
    >>> class A():
    ...     def f(self):
    ...         return 10
    ...
    >>> from skmine.callbacks import CallBacks
    >>> stack = list()
    >>> callbacks = CallBacks(f=stack.append)
    >>> a = A()
    >>> callbacks(a)
    >>> a.f()
    10
    >>> stack
    [10]
    """

    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self._check()

    def _check(self):
        # TODO : inspect source code from callbacks and check no assigment on inner self
        for v in self.values():
            if not callable(v):
                raise TypeError(f"values must be callables, found {type(v)}")
            if has_self_assigment(v):  # TODO : only allow lambdas or builtins ?
                raise ValueError("callbacks should not modify `self` attributes")

    def _frozen(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self)} is immutable")

    __setitem__ = _frozen
    update = _frozen

    def __call__(self, miner):
        miner_methods = [
            f_name for f_name in dir(miner) if callable(getattr(miner, f_name))
        ]
        for f_name in self.keys():
            if not f_name in miner_methods:
                raise ValueError(
                    f"{f_name} found in callbacks while there is not corresponding function"
                )

        for callback_name, callback in self.items():
            new_meth = post(miner, callback_name, callback)
            # TODO : lock assignement to "once for all"
            # re-executing a cell in a notebook can lead to callbacks being called many times
            setattr(miner, callback_name, new_meth)


def _print_positive_gain(self, data_size, model_size, *_):
    diff = (self.model_size_ + self.data_size_) - (data_size + model_size)
    if diff > 0.01:
        print(
            "data size : {:.2f} | model size : {:.2f}".format(data_size, model_size)
        )

def _print_candidates_size(self, candidates):
    print("{} new candidates considered".format(len(candidates)))


mdl_prints = CallBacks(
    evaluate=_print_positive_gain, generate_candidates=_print_candidates_size
)
mdl_prints.__doc__ = """
Base callback for miners which inherit the :class:`skmine.base.MDLOptimizer`

Prints data size and model size when compression has improved,
only if ``verbose`` is set to True for the miner to attach.

Examples
--------
>>> from skmine.callbacks import mdl_prints
>>> from skmine.base import MDLOptimizer
>>> class MyMDLMiner(MDLOptimizer):
>>>    def __init__(self):
...        self.codetable_ = dict()
...    def generate_candidates(self):
...        return [(2,), (2, 3), (2, 4)]
...    def evaluate(self): pass

>>> miner = MyMDLMiner()
>>> mdl_prints(miner)
>>> miner.generate_candidates()
3 new candidates considered
[(2,), (2, 3), (2, 4)]
"""
