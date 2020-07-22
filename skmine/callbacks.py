"""
Callback API for scikit-mine
"""
from inspect import signature

def _get_params(fn):
    assert callable(fn)
    try:
        sig = signature(fn)
        params = set(sig.parameters.keys())
    except ValueError:
        params = set()
    return params

def post(self, func_name, callback):
    """decorator, call callback on returned values, after main function"""
    func = getattr(self, func_name)
    assert callable(func)
    callback_params = _get_params(callback)
    def _(*args, **kwargs):
        res = func(*args, **kwargs)
        if 'self' in callback_params and len(callback_params) > 1:
            callback(self, res)     # eg. def f(self, x): print(self.b, x)
        elif callback_params == {'self'}:
            callback(self)           # eg. list.append
        else:
            callback(res)            # eg. list.append
        return res
    return _


class CallBacks(dict):
    """
    A collection of callbacks

    Works by defining functions to be called after the execution of the function they target

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
    >>> _ = callbacks(a)
    >>> a.f()
    10
    >>> stack
    [10]
    """
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self._check()

    def _check(self):
        assert all(map(callable, self.values()))

    def _frozen(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self)} is immutable")

    __setitem__ = _frozen
    update = _frozen

    def __call__(self, miner):
        miner_methods = [f_name for f_name in dir(miner) if callable(getattr(miner, f_name))]
        for f_name in self.keys():
            if not f_name in miner_methods:
                raise ValueError(
                    f"{f_name} found in callbacks while there is not corresponding function"
                )

        for callback_name, callback in self.items():
            new_meth = post(miner, callback_name, callback)
            setattr(miner, callback_name, new_meth)

        return miner
    