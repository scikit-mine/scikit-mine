# TODO : visitor


def get_occs(node, tau=0):
    dists = [0] + node.children_dists
    for shift in range(tau, (node.r * node.p) + tau, node.p):
        dist_acc = 0
        for dist, child in zip(dists, node.children):
            dist_acc += dist
            if isinstance(child, Node):
                yield from get_occs(child, tau=shift + dist_acc)
            else:
                yield shift + dist_acc, child  # leaf


class Node:
    """
    base Node class for periodic elements

    Parameters
    ----------
    r: int
        number of repetitions for this node
    p: int
        period, inter occurence delay for this node

    """

    def __init__(self, r, p, *, children: list = list(), children_dists: list = list()):
        if not len(children) - 1 == len(children_dists):
            raise ValueError(
                "There should be exactly `|children| - 1` inter-child distances"
            )
        self.r = int(r)  # number of repetitions
        self.p = int(p)  # period of time
        self.children_dists = children_dists
        self.children = children


class Tree(Node):
    """
    Periodic tree structure

    A tree is a pointer to the root node. To this regard, the root has the same
    attributes as a Node, those being:
     - r: number of repetitions
     - p: period
     - children: list of children nodes
     - children_dists: distances between brothers and sisters

    The tree datastructure also holds specific attributes, like:
     - tau: the starting offset, i.e the position of the very first event described by the tree
     - E: a collections of shift corrections for events described by the tree
    """

    def __init__(self, tau, r, p, *args, **kwargs):
        super(Tree, self).__init__(r, p, *args, **kwargs)
        self.tau = tau
        self.E = list()  # TODO, np.array ??

    def get_occs(self):
        """
        unfold the tree and retrieve all occurences of the events it describes
        """
        return get_occs(self, tau=self.tau)

    def tolist(self):
        """Returns the tree as a list of (occurence, event) pairs"""
        return list(self.get_occs())

    def __str__(self):
        """
        get the expression for this tree
        """
        pass
