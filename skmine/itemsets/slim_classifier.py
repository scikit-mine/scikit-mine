import numpy as np
from functools import reduce
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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
        return {"no_validation": True}

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
        self._validate_data(X, y, reset=True, validate_separately=False, force_all_finite=False,
                            accept_sparse=False, ensure_2d=False, ensure_min_samples=0, dtype=list)
        self.classes_ = np.unique(y)
        self.classes_X_ = []
        self.models_ = []

        for c in self.classes_:
            transactions_classes = [transaction for transaction, target in zip(X, y) if target == c]
            self.classes_X_.append(transactions_classes)
            self.models_.append(SLIM(items=self.items))

        for model, data in zip(self.models_, self.classes_X_):
            model.fit(data)

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

        # self.models_scores = {i: model.decision_function(X).values for i, model in enumerate(self.models_)}
        self.models_scores = np.vstack([model.decision_function(X).values for model in self.models_]).T

        # for i in range(len(X)):
        #     scores = [model[i] for model in self.models_scores.values()]
        #     # best_index = scores.index(reduce(lambda x, y: x if abs(y) > abs(x) else y, scores))
        #     # lowest absolute value coincide with maximum value for negative
        #     best_index = np.argmax(scores)
        #     predictions.append(self.classes_[0] if best_index is None else self.classes_[best_index])
        # return np.array(predictions)
        return self.classes_[self.models_scores.argmax(axis=1)]

    def __copy__(self):
        return SlimClassifier(items=self.items, pruning=self.pruning)

#
# if __name__ == '__main__':
#     from skmine.datasets.fimi import fetch_iris
#     from sklearn.preprocessing import KBinsDiscretizer
#     from sklearn.pipeline import Pipeline
#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder, LabelBinarizer
#     from sklearn.metrics import confusion_matrix
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split
#     import pandas as pd
#     from skmine.itemsets import SLIM
#
#
#     # # X, y = fetch_iris(return_y=True)
#     # # y = y - 17
#     # X, y = load_iris(return_X_y=True)
#     # nbins = 10
#     # encode = 'onehot-dense'
#     # # without return_y=True, the method would have returned the whole dataset in one variable
#     # print("X shape:", X.shape, "y shape:", y.shape)
#     #
#     # est = KBinsDiscretizer(n_bins=nbins, encode=encode, strategy='uniform')
#     # Xt = est.fit_transform(X)
#     # print(Xt[:5])
#     # print("Xt.shape", Xt.shape)
#     #
#     # (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=True)
#     # print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
#     # print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
#     # # You can pass in parameter of your classifier the set of your items.
#     # # This will improve its performance especially on small data sets like iris.
#     # # items = set(item for transaction in X for item in transaction)
#     #
#     # clf = Pipeline([
#     #     ('discretizer', KBinsDiscretizer(n_bins=nbins, encode=encode, strategy='uniform')),
#     #     ('OvRslim',  OneVsRestClassifier(SLIM())), # SlimClassifier(items=np.unique(Xt).tolist()) ),
#     # ])
#     # #
#     # # # clf = SlimClassifier(items=items)
#     # clf.fit(X_train, y_train)
#     # print("SCORE ", clf.score(X_test, y_test) * 100, ' %')
#     # #
#     # y_pred = clf.predict(X_test)
#     # conf_mat = confusion_matrix(y_test, y_pred)
#     # print(conf_mat)
#     # print('*' * 60, '\n OneVsRestClassifier fit ')
#
#     #
#     # class CustomMultiLabelBinarizer(BaseEstimator, TransformerMixin):
#     #     def fit(self, X, y=None):
#     #         return self
#     #
#     #     def transform(self, X):
#     #         return MultiLabelBinarizer(sparse_output=False).fit_transform(X)
#     #
#     #
#     # slim_onehot = Pipeline([
#     #     ('transaction_encoder', CustomMultiLabelBinarizer()),
#     #     ('OvRslim', OneVsRestClassifier(SLIM())),
#     # ])
#     #
#     # binar = CustomMultiLabelBinarizer()
#     #
#     # ovr = OneVsRestClassifier(SLIM())
#     # slim_onehot.fit(X_train, y=y_train)
#     # y_pred = slim_onehot.predict(X_test)
#     # conf_mat = confusion_matrix(y_test, y_pred)
#     # # print(slim_onehot.decision_function(binar.fit_transform(X_test)))  # , columns=ovr.classes_)
#     #
#     # print("SCORE ", slim_onehot.score(X_test, y_test))
#     # print(conf_mat)
#
#     ###########################################################################################################
#
#     ##########################################################################################################
#
#     class TransactionEncoder(MultiLabelBinarizer):  # pandas DataFrames are easier to read ;)
#         def transform(self, X):
#             _X = super().transform(X)
#             return pd.DataFrame(data=_X, columns=self.classes_)
#
#     #
#
#     def print_ovr_prop(ovr_ex):
#         print("ovr_ex.label_binarizer_", ovr_ex.label_binarizer_)
#         print("ovr_ex.n_features_in_", ovr_ex.n_features_in_)
#         print("ovr_ex.estimators_", ovr_ex.estimators_)
#         print("ovr_ex.n_classes_", ovr_ex.n_classes_)
#
#
#     transactions = [
#         ['milk', 'bananas'],
#         ['tea', 'New York Times', 'El Pais'],
#         ['New York Times'],
#         ['El Pais', 'The Economist'],
#         ['milk', 'tea'],
#         # ['croissant', 'tea'],
#         # ['croissant', 'chocolatine', 'milk'],
#     ]
#     new_transaction = [
#         ['bananas', 'tea'],
#         ['El Pais', 'tea']
#     ]
#     new_labels = [
#         'foodstore',
#         'newspaper'
#     ]
#     labels = [
#         'foodstore',
#         'newspaper',
#         'newspaper',
#         'newspaper',
#         'foodstore',
#         # 'bakery',
#         # 'bakery',
#     ]
#     print('*' * 60, '\n DATA preprocessing ')
#
#     print("INITIAL transactions :", *transactions, sep='\n')
#     print("Labels :\n", pd.Series(labels))
#     print("nb_class :", len(np.unique(labels)))
#     items = set(item for t in transactions for item in t)
#     print("All items : ", items)
#     te = TransactionEncoder()
#     te.fit(transactions)
#
#     # binar = MultiLabelBinarizer(sparse_output=False)
#     # X = binar.fit_transform(transactions)
#     X = te.transform(transactions)
#     new_X = te.transform(new_transaction)
#     print("transactions one hot :\n", X)
#     print("new transactions :", *new_transaction, sep='\n')
#     print("new_transaction one hot :\n", new_X)
#
#     # ====================================================
#     # SLIM 2 classe
#     # ==========================================================
#     # slim = SLIM()
#     # res = slim.fit(transactions).transform()
#     # print("code table res \n", res)
#     # codes = slim.decision_function(new_X)
#     # print(pd.DataFrame([pd.Series(new_transaction), codes], index=['transaction', 'distance']).T)
#     #
#     # # print('*'*60, '\n SLIM fit_transform ')
#     # # slim = SLIM(items)
#     # # res = slim.fit(X)
#     # # print(res)
#     # # print(pd.DataFrame(slim.decision_function(new_X))) #, columns=binar.classes_))
#     # # D = transactions
#     #
#
#     print('*' * 60, '\n OneVsRestClassifier fit ')
#     ovr = OneVsRestClassifier(SLIM(), verbose=0)
#     ovr.fit(X, y=labels)
#     # print(ovr.estimators_, )
#     print("fitted classes: ", ovr.classes_)
#
#     if len(ovr.classes_) > 2:
#         print(pd.DataFrame(ovr.decision_function(new_X), columns=ovr.classes_))
#     else:
#         print("ovr decision\n", ovr.decision_function(new_X))
#     y_test_ovr = ovr.predict(new_X)
#     print("Predictions OneVsRestClassifier :", y_test_ovr)
#
#     print('*' * 60, '\n SlimClassifier fit ')
#     clf = SlimClassifier(items=items)
#     clf.fit(transactions, labels)
#     print("fitted classes: ", clf.classes_, "\n")
#
#     y_test_slim = clf.predict(new_transaction)
#     print(pd.DataFrame(clf.models_scores, columns=clf.classes_))
#     print("Predictions SlimClassifier :", y_test_slim)
#     print('-' * 60, "\n GT new_labels", new_labels)
#
#     # print_ovr_prop(ovr)
#     # y_pred = ovr.predict(D)
#     # print(y_pred)
#     # conf_mat = confusion_matrix(y, y_pred)
#     # print(conf_mat)
#
#     # import numpy as np
#     # #erreur Predict_proba OvR https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/multiclass.py#L455
#     # slim = SLIM(items)
#     # res = slim.fit(X)
#     # Y0 = slim.predict_proba(X)
#     # print(Y0.toarray())
#     # print("slim.predict_proba(X).shape", Y0.shape)
#     # Y = np.array([Y0[:, 1]]).T
#     #
#     # print("Y", Y)
#     # print("Y.shape", Y.shape)
#
#     # from sklearn import datasets
#     # from sklearn.neighbors import KNeighborsClassifier
#     #
#     # # Load the Iris dataset
#     # iris_X, iris_y = datasets.load_iris(return_X_y=True)  # Split Iris dataset into train/test sets randomly
#     # np.random.seed(0)
#     # indices = np.random.permutation(len(iris_X))
#     # iris_X_train = iris_X[indices[:-10]]
#     # iris_y_train = iris_y[indices[:-10]]
#     # iris_X_test = iris_X[indices[-10:]]
#     # iris_y_test = iris_y[indices[-10:]]  # Instantiate and fit a KNeighbors classifier
#     # knn = KNeighborsClassifier()
#     # knn.fit(iris_X_train, iris_y_train)
#     # predictions = knn.predict(iris_X_test)
#     # print(predictions)
#     # predictions = knn.predict_proba(iris_X_test)
#     # print(predictions)
