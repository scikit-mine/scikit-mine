import copy
from itertools import chain, combinations, groupby, zip_longest

import numpy as np
from sortedcontainers import SortedKeyList

from .cycles import PeriodicCycleMiner, extract_triples, merge_triples


def get_occs(node, tau=0):
    """
    get occurences covered by a node (or tree)
    """
    dists = [0] + node.children_dists
    for shift in np.arange(tau, (node.r * node.p) + tau, node.p):
        dist_acc = 0
        for dist, child in zip(dists, node.children):
            dist_acc += dist
            if isinstance(child, Node):
                yield from get_occs(child, tau=shift + dist_acc)
            else:
                yield shift + dist_acc, child  # leaf


def prefix_visitor(tree):
    """visit tree in prefix order"""

    def _inner(node):
        for child in node.children:
            yield child
            if isinstance(child, Node):
                yield from _inner(child)

    yield tree
    yield from _inner(tree)


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
        if children and (not len(children) - 1 == len(children_dists)):
            raise ValueError(
                "There should be exactly `|children| - 1` inter-child distances"
            )
        self.r = int(r)  # number of repetitions
        self.p = float(p)  # period of tim
        self.children_dists = list(children_dists)
        self.children = list(children)

    def size(self):
        return sum((1 for _ in prefix_visitor(self)))

    __len__ = size

    def __eq__(self, o):
        if not isinstance(o, Node):
            return False
        return (
            self.r == o.r
            and self.p == o.p
            and all(
                (a == b for a, b in zip_longest(self.children_dists, o.children_dists))
            )
            and all((a == b for a, b in zip_longest(self.children, o.children)))  # rec
        )


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

    def get_internal_nodes(self):
        return filter(lambda n: isinstance(n, Node), prefix_visitor(self))

    def __str__(self):
        """
        get the expression for this tree
        """
        pass

    def to_node(self):
        """remove `tau` and `E` from the current object, return a new instance of Node"""
        return Node(
            self.r, self.p, children=self.children, children_dists=self.children_dists
        )


class Forest(SortedKeyList):  # TODO
    """A forest is a collection of trees. Trees are sorted by their `tau`s"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._key = lambda t: (t.tau, len(t))

    append = SortedKeyList.add


def combine_vertically(H: list):
    """
    combine trees verically, by detecting cycles on their `tau`s
    """
    V_prime = list()
    H = sorted(H, key=lambda e: e.tau)
    while H:  # for each distinc tree
        Tc = H[0]
        C = [t for t in H if t == Tc]
        taus = np.array([_.tau for _ in C])
        cycles_tri = extract_triples(taus)  # TODO : pass `l_max`
        cycles_tri = merge_triples(cycles_tri)
        for cycle_batch in cycles_tri:
            p_vect = np.median(np.diff(cycle_batch, axis=1), axis=1)
            r = cycle_batch.shape[1]
            for tau, p in zip(cycle_batch[:, 0], p_vect):
                # create a new tree to make sure we don't mistankenly
                # manipulate references on the root
                K = Tree(tau, r=r, p=p, children=[Tc.to_node()])
                # TODO : check cost (line 8 from algorithm 4)
                V_prime.append(K)
                H = [
                    _ for _ in H if _ not in C
                ]  # FIXME : this differs from the original paper
        else:
            break

    return V_prime


def grow_horizontally(*trees):
    """Grow trees horizontally"""
    p = list(set((_.p for _ in trees)))
    if len(p) != 1:
        raise ValueError("all trees should have same p to grow horizontally")
    p = p[0]
    r = min((_.r for _ in trees))
    children_dists = [b.tau - a.tau for a, b in zip(trees, trees[1:])]
    children = [
        t if any(map(lambda c: isinstance(c, Node), t.children)) else t.children[0]
        for t in trees
    ]
    tau = trees[0].tau
    return Tree(tau, r, p, children=children, children_dists=children_dists)


def combine_horizontally(V: list):
    H_prime = list()
    G = list()
    C = [
        (Pa, Pb)
        for Pa, Pb in combinations(V, 2)
        if Pa.p == Pb.p and Pb.tau <= Pa.tau + Pa.p
    ]
    for Pa, Pb in C:
        K = grow_horizontally(Pa, Pb)
        # TODO : evaluate len of K here
        H_prime.append(K)
        G.append((Pa, Pb))

    cliques = groupby(G, key=lambda _: _[0].p)
    for _, clique in cliques:
        stack = set()
        flat_clique = list()
        for t in chain(*clique):
            if t.tau not in stack:
                flat_clique.append(t)
                stack.add(t.tau)

        clique_T = grow_horizontally(*flat_clique)
        H_prime.insert(0, clique_T)

    return H_prime


class PeriodicPatternMiner:
    """
    Mining Periodic Pattern with a MDL criterion

    Implementation of periodic tree mining.

    This first extract cycles from the input data, and then combine these cycles
    into more complex tree structures.

    A tree is defined as a 3-tuple of the form
    :math: `\tau`, `C`, `E`

    See Also
    --------
    skmine.periodic.PeriodicCycleMiner
    """

    def __init__(self, max_length=100):
        # TODO : pass instance of PeriodicCycleMiner, check is_fitted
        self.cycle_miner = PeriodicCycleMiner(max_length=max_length)
        self.forest = list()

    def _prefit(self, D):
        cycles = self.cycle_miner.fit_discover(D)
        singletons = list()
        for o in cycles.itertuples():
            t = Tree(o.start, r=o.length, p=o.period, children=[o.Index[0]])
            singletons.append(t)

        singletons = sorted(singletons, key=lambda s: s.tau)

        return singletons

    def fit(self, D):
        """
        Discover periodic patterns (in the form of trees) from a sequence of event `D`

        This iteratively refines the set of trees by successive vertical/horizontal
        combinations, starting from single-node trees describing `cycles`.

        The resulting model is a list of periodic trees.
        """
        singletons = self._prefit(D)
        # singletons = [Tree(tau=)]
        H = copy.deepcopy(singletons)  # list of horizontal combinations
        V = copy.deepcopy(singletons)  # list of vertical combinations

        C = list()

        while V:  # TODO while H or V
            V_prime = combine_vertically(H)
            H_prime = combine_horizontally(V)
            V = V_prime
            H = H_prime
            C += H + V

        # TODO P = greedy_cover(C, S), return P
        self.forest = C
        return self
