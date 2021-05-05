"""
Periodic trees
"""
import dataclasses
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sortedcontainers import SortedKeyList

from ..utils import bron_kerbosch
from .cycles import PeriodicCycleMiner, extract_triples, merge_triples

L_PARENTHESIS = -np.log2(
    1 / 3
)  # length of either `(` or `)` when encoding tree patterns


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


def encode_leaves(node, event_freqs: dict):
    """MDL cost for leaves

    Parameters
    ----------
    event_freqs: dict[object, float]
        mapping of frequencies for every event in the input data
    """
    if isinstance(node, Node):
        return (
            L_PARENTHESIS
            + sum([encode_leaves(child, event_freqs) for child in node.children])
            + L_PARENTHESIS
        )
    freq = event_freqs[node]
    return -np.log2(freq / 3)


@dataclasses.dataclass(frozen=True)
class Node:
    """
    base Node class for periodic elements

    Parameters
    ----------
    r: int
        number of repetitions for this node
    p: int
        period, inter occurence delay for this node
    children: list
        list of children
    children_dists: list
        list of inter node distances between children
    """

    r: int
    p: float
    children: list = dataclasses.field(default_factory=list, hash=False)  # tuple ?
    children_dists: list = dataclasses.field(default_factory=list, hash=False)

    def __post_init__(self):
        if self.children and (not len(self.children) - 1 == len(self.children_dists)):
            raise ValueError(
                "There should be exactly `|children| - 1` inter-child distances"
            )

    def size(self):
        """returns the number of nodes in the tree"""  # TODO : reuse __len__ from children
        return sum((1 for _ in prefix_visitor(self)))

    @property
    def N(self):
        return len(get_occs(self))  # TODO : there is a more efficient way

    to_dict = dataclasses.asdict
    to_tuple = dataclasses.astuple
    __len__ = size

    def __eq__(self, other):  # TODO remove
        return isinstance(other, Node) and self.to_tuple() == other.to_tuple()


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

    def __init__(self, tau, r, p, E=None, *args, **kwargs):
        super(Tree, self).__init__(r, p, *args, **kwargs)
        self.tau = tau
        if E is None:
            self.E = np.array([0] * (len(self.children) - 1))
        else:
            assert hasattr(E, "__len__")
            self.E = np.array(E)

    def get_occs(self):
        """
        unfold the tree and retrieve all occurences of the events it describes
        """
        return get_occs(self, tau=self.tau)

    def to_list(self):
        """Returns the tree as a list of (occurence, event) pairs"""
        return list(self.get_occs())

    def get_internal_nodes(self):
        """yield interal nodes"""
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


def grow_horizontally(*trees, presort=False):
    """Grow trees horizontally"""
    if presort:
        trees = sorted(trees, key=lambda t: t.tau)
    p = np.median([t.p for t in trees])
    r = min((_.r for _ in trees))
    children_dists = [b.tau - a.tau for a, b in zip(trees, trees[1:])]
    children = [
        t if any(map(lambda c: isinstance(c, Node), t.children)) else t.children[0]
        for t in trees
    ]
    tau = trees[0].tau
    E = trees[0].E  # FIXME ?
    return Tree(tau, r, p, E, children=children, children_dists=children_dists)


def combine_horizontally(V: list):
    H_prime = list()
    G = defaultdict(list)
    C = [(Pa, Pb) for Pa, Pb in combinations(V, 2) if Pb.tau <= Pa.tau + Pa.p]

    for Pa, Pb in C:
        K = grow_horizontally(Pa, Pb)
        # TODO : evaluate len of K here
        H_prime.append(K)
        G[Pa].append(Pb)
        G[Pb].append(Pa)

    cliques = bron_kerbosch(G)
    for clique in cliques:
        clique_T = grow_horizontally(*clique, presort=True)
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
        self.forest = Forest()

    def _prefit(self, D):
        cycles = self.cycle_miner.fit_discover(D, shifts=True)
        singletons = Forest()
        for o in cycles.itertuples():
            t = Tree(o.start, r=o.length, p=o.period, children=[o.Index[0]], E=o.dE)
            singletons.append(t)

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
        H = singletons  # list of horizontal combinations
        V = singletons  # list of vertical combinations

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

    def discover(self):
        cols = ["tau", "root", "shifts"]
        data = [(t.tau, t.to_node(), t.shifts) for t in self.forest]
        return pd.DataFrame(data, columns=cols)
