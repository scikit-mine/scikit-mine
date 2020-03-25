from itertools import chain


def describe_itemsets(D):
    """
    D: pd.Series
    """
    return dict(
        nb_items=len(set(chain(*D))),
        avg_transaction_size=D.map(len).mean(),
        nb_transactions=D.shape[0]
    )
