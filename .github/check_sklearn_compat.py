"""
Compatibility checks between sklearn and skmine

Skmine is dedicated to exploratory data analysis.
To this regard, some checks are considered as non-mandatory and are thus avoided.
"""
import inspect
import sys

import sklearn
from sklearn.utils.estimator_checks import check_estimator

import skmine.itemsets
import skmine.preprocessing

MODULES = [
    skmine.itemsets,
    skmine.preprocessing,
]

EXCLUDED_CHECKS = [
    "check_no_attributes_set_in_init",
    "check_estimator_sparse_data",
    "check_estimators_pickle",
    "check_estimators_dtypes",
    "check_methods_subset_invariance",
    "check_dict_unchanged",
    "check_fit_idempotent",
    "check_transformer_general",
]

OK = "\x1b[42m[ OK ]\x1b[0m"
FAIL = "\x1b[41m[FAIL]\x1b[0m"


def verify(e):
    _, est = e
    fit_meth = getattr(est, "fit", None)
    pred_meth = getattr(est, "decision_function", None)
    transform_meth = getattr(est, "transform", None)
    return callable(fit_meth) and (callable(pred_meth) or callable(transform_meth))


if __name__ == "__main__":
    ret_code = 0
    for module in MODULES:
        clsmembers = inspect.getmembers(module, inspect.isclass)
        estimators = filter(verify, clsmembers)
        for est_name, est in estimators:
            # from sklearn 0.23 check_estimator takes an instance as input
            obj = est() if sklearn.__version__[:4] >= "0.23" else est
            checks = check_estimator(obj, generate_only=True)
            for arg, check in checks:
                check_name = check.func.__name__  # unwrap partial function
                if check_name in EXCLUDED_CHECKS:
                    continue
                desc = "{} === {}".format(est_name, check_name)
                try:
                    check(arg)
                    print(OK, desc)
                except Exception as e:
                    print(FAIL, desc, e)
                    ret_code = 1

    sys.exit(ret_code)
