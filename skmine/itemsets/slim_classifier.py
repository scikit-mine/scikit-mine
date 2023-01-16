import numpy as np
from functools import reduce
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from skmine.itemsets.slim import SLIM


class SlimClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier using the SLIM compression algorithm. Works for binary and multi-class problems.

    This classifier uses one SLIM instance per class in the database, resulting in a code table per class.
    To classify a transaction, we simply assign the class belonging to the code table that provides the minimal encoded
    length for the transaction.

    Parameters
    ----------
    items: set, default=None
        The list of items in the complete dataset not only the training set. This improves the accuracy of the model.
        Without this set of items, the classifier works but is less good in particular with small datasets.

    pruning: bool, default=False
        Indicates whether each SLIM classifier enables pruning

    Attributes
    ----------
    classes_ : array-like
        All the unique classes

    models_ : list
        A list of SLIM instances corresponding to *classes_*

    classes_X_ : list
        A list where each element is a subset of X and each element contains the transactions of X associated
        with the class from *classes_*  of the same index
    """

    def __init__(self, items=None, pruning=False):
        self.items = items
        self.pruning = pruning

    def _more_tags(self):
        return {
            "no_validation": True,
        }

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X: iterable, {array_like}
            containing n_transactions containing themselves n_items

        y: array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            An instance of the estimator
        """
        self._validate_data(X, y, reset=True, validate_separately=False,
                            force_all_finite=False, accept_sparse=False, ensure_2d=False, ensure_min_samples=0,
                            dtype=list)
        self.classes_ = np.unique(y)
        self.classes_X_ = []
        self.models_ = []

        for c in self.classes_:
            transactions_classes = [transaction for transaction, target in zip(X, y) if target == c]
            self.classes_X_.append(transactions_classes)
            self.models_.append(SLIM(items=self.items))

        for model, data in zip(self.models_, self.classes_X_):
            model.fit(data)
        # self.n_features_in_ = X.shape[-1] if not isinstance(X, list) else len(X)
        if len(y) != len(X):
            raise ValueError(f'TA TAA YOYOYO')

        return self

    def predict(self, X):
        """Perform classification on samples in X

        Parameters
        ----------
        X : iterable containing n_transactions containing themselves n_items

        Returns
        -------
        y_pred : np.array of shape (n_samples,)
            Class labels for samples in X
        """
        check_is_fitted(self, "classes_")
        # n_features_in_ = X.shape[-1] if not isinstance(X, list) else len(X)

        # if n_features_in_ != self.n_features_in_:
        #     raise ValueError(f'Shape of input predict {n_features_in_} is different from what was seen in `fit` {self.n_features_in_}')

        # if self.classes_ is None:
        #     raise ValueError("fit must be called first.")

        models_scores = {i: model.decision_function(X).values for i, model in enumerate(self.models_)}
        predictions = []

        for i in range(len(X)):
            scores = [model[i] for model in models_scores.values()]
            best_index = scores.index(reduce(lambda x, y: x if abs(y) > abs(x) else y, scores))
            predictions.append(self.classes_[0] if best_index is None else self.classes_[best_index])

        return np.array(predictions)

    def __copy__(self):
        return SlimClassifier(items=self.items, pruning=self.pruning)


if __name__ == '__main__':
    # from skmine.datasets.fimi import fetch_iris
    # #
    # X, y = fetch_iris(return_y=True)  # without return_y=True, the method would have returned the whole dataset in one variable
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)
    # print(X.head())
    # print(y.head())
    # from sklearn.model_selection import train_test_split
    #
    # (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True)
    # print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
    # # You can pass in parameter of your classifier the set of your items.
    # # This will improve its performance especially on small data sets like iris.
    # items = set(item for transaction in X for item in transaction)
    #
    # clf = SlimClassifier(items=items)
    # clf.fit(X_train, y_train)
    # print("SCORE ", clf.score(X_test, y_test))
    # from sklearn.metrics import confusion_matrix
    #
    # y_pred = clf.predict(X_test)
    # conf_mat = confusion_matrix(y_test, y_pred)
    # print(conf_mat)

    import pandas as pd
    from skmine.itemsets import SLIM
    from sklearn.preprocessing import MultiLabelBinarizer

    #
    # class TransactionEncoder(MultiLabelBinarizer):  # pandas DataFrames are easier to read ;)
    #     def transform(self, X):
    #         _X = super().transform(X)
    #         return pd.DataFrame(data=_X, columns=self.classes_)
    #

    # transactions = [
    #     ['bananas', 'milk'],
    #     ['milk', 'bananas', 'cookies'],
    #     ['cookies', 'butter', 'tea'],
    #     ['tea'],
    #     ['milk', 'bananas', 'tea'],
    # ]
    # # # te = TransactionEncoder()
    # # # D = te.fit(transactions).transform(transactions)
    # print(transactions)
    # binar = MultiLabelBinarizer(sparse_output=False)
    # D = binar.fit_transform(transactions)
    # print(pd.DataFrame(data=D, columns=binar.classes_))
    # slim = SLIM()
    # res = slim.fit(transactions).transform(transactions)
    # # print(res)
    # new_transactions = [
    #     ['bananas', 'milk'],
    #     ['milk', 'sirup', 'cookies'],
    #     ['butter', 'tea'],
    #     ['tea'],
    #     ['milk', 'bananas', 'tea'],
    # ]
    # binar_new = MultiLabelBinarizer()
    #
    # new_D = binar_new.fit_transform(transactions) # te.transform(new_transactions)
    # print(pd.DataFrame(data=new_D, columns=binar_new.classes_))
    # codes = slim.decision_function(new_D)
    # print(pd.DataFrame([pd.Series(new_transactions), codes], index=['transaction', 'distance']).T)
    from sklearn.multiclass import OneVsRestClassifier

    # from sklearn.pipeline import Pipeline
    # #
    # slim_onehot = Pipeline([
    #     ('transaction_encoder', TransactionEncoder),
    #     ('slim', SLIM()),
    # ])
    transactions = [
        ['milk', 'bananas'],
        ['tea', 'New York Times', 'El Pais'],
        ['New York Times'],
        ['El Pais', 'The Economist'],
        ['milk', 'tea'],
        # ['croissant', 'tea'],
        # ['croissant', 'chocolatine', 'milk'],
    ]
    new_D = [['milk', 'tea'],
             'El Pais','The Economist' ]
    y = [
        'foodstore',
        'newspaper',
        'newspaper',
        'newspaper',
        'foodstore',
        # 'bakery',
        # 'bakery',
    ]
    print(transactions)
    binar = MultiLabelBinarizer(sparse_output=False)
    D = binar.fit_transform(transactions)

    print(D) # pd.DataFrame(data=D, columns=binar.classes_))
    print('*'*60, '\n SLIM fit_transform ')
    slim = SLIM()
    res = slim.fit(D)
    print(res)
    print(pd.DataFrame(slim.decision_function(D))) #, columns=binar.classes_))

    # def print_ovr_prop(ovr_ex):
    #     print("ovr_ex.label_binarizer_", ovr_ex.label_binarizer_)
    #     print("ovr_ex.n_features_in_", ovr_ex.n_features_in_)
    #     print("ovr_ex.estimators_", ovr_ex.estimators_)
    #     print("ovr_ex.n_classes_", ovr_ex.n_classes_)
    # te = TransactionEncoder()
    # D = te.fit(transactions).transform(transactions)
    # print(D)
    # D = transactions
    print('*'*60, '\n OneVsRestClassifier fit ')

    ovr = OneVsRestClassifier(slim, verbose=0)
    ovr.fit(D, y=y)
    # print(ovr.estimators_, )
    ovr.decision_function(D) # , columns=ovr.classes_)
    #
    # print_ovr_prop(ovr)
    #
    # y_pred = ovr.predict(D)
    # print(y_pred)
    # #
    # from sklearn.metrics import confusion_matrix
    #
    # conf_mat = confusion_matrix(y, y_pred)
    # print(conf_mat)




