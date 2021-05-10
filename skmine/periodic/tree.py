"""
Periodic trees
"""
import array
import dataclasses
from collections import defaultdict
from functools import partial
from itertools import chain, combinations, count

import numpy as np
import pandas as pd
from sortedcontainers import SortedKeyList

from ..utils import bron_kerbosch
from .cycles import PeriodicCycleMiner, extract_triples, merge_triples

L_PARENTHESIS = -np.log2(
    1 / 3
)  # length of either `(` or `)` when encoding tree patterns

shift_array = partial(array.array, "i")  # shifts are signed integers


def get_occs(node, E=None, tau=0, sort=True, r=None):
    """
    get occurences covered by a node (or tree)
    """
    if not E:
        E = np.zeros(node._n_occs, dtype=int)

    # E = np.cumsum(E)
    if not hasattr(E, "typecode"):
        E = shift_array(E)
    # assert len(E) == node._n_occs  TODO

    def _get_occs(node, acc, E, tau=0, dist_acc=0):
        dists = np.cumsum([0] + node.children_dists)
        inters = np.arange(tau, (node.r * node.p) + tau, node.p)
        for idx, inter in zip(count(), inters):
            dist_acc = 0
            for idy, dist, child in zip(count(), dists, node.children):
                if isinstance(child, Node):
                    new_tau = inter + dist + E[len(acc)]
                    _get_occs(child, acc, E, tau=new_tau)
                elif r is not None and len(acc) >= r:
                    return
                else:
                    e = E[len(acc)]
                    dist_acc += e
                    if idx + idy == 0:
                        dist_acc = 0
                    occ = (inter + dist + dist_acc, child)
                    acc.append(occ)  # leaf

    acc = list()
    _get_occs(node, acc, E, tau)
    if sort:
        acc = sorted(acc)
    return acc


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


@dataclasses.dataclass
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
    _size: int = dataclasses.field(init=False, repr=False)
    _n_occs: int = dataclasses.field(init=False, repr=False)
    children: list = dataclasses.field(default_factory=list, hash=False)  # tuple ?
    children_dists: list = dataclasses.field(default_factory=list, hash=False)

    def __post_init__(self):
        if self.children and (not len(self.children) - 1 == len(self.children_dists)):
            raise ValueError(
                "There should be exactly `|children| - 1` inter-child distances"
            )

        d_sum = sum(self.children_dists)
        if d_sum > self.p:
            raise ValueError(
                f"""
                sum of inter children distances is equal to {d_sum}.
                `p`, which is equal to {self.p}, must be greater.
                """
            )

        self._size = 1 + sum(
            (c._size if isinstance(c, Node) else 1 for c in self.children)
        )
        self._n_occs = self.r * sum(
            (c._n_occs if isinstance(c, Node) else 1 for c in self.children)
        )

    to_dict = dataclasses.asdict
    to_tuple = dataclasses.astuple

    def __len__(self):
        return self._size

    def __eq__(self, other):
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
            self.E = shift_array([0] * self._n_occs)
        else:
            assert hasattr(E, "__len__") and len(E) == self._n_occs
            self.E = shift_array(E)

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


def grow_horizontally(*trees, presort=True):
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
    T = Tree(tau, r, p, children=children, children_dists=children_dists)

    occs1 = sorted(chain(*(get_occs(t, tau=t.tau, E=t.E, r=r) for t in trees)))
    occs2 = get_occs(T, tau=tau)  # cover instances with a perfect tree
    E = [b[0] - a[0] for a, b in zip(occs1, occs2)]  # perfect covering VS real occs
    T.E = E
    return T


def combine_horizontally(V: list):
    H_prime = list()
    G = defaultdict(list)
    C = [(Pa, Pb) for Pa, Pb in combinations(V, 2) if Pb.tau <= Pa.tau + Pa.p]

    for Pa, Pb in C:
        K = grow_horizontally(Pa, Pb)
        # TODO : evaluate len of K here
        H_prime.append(K)
        G[id(Pa)].append(id(Pb))
        G[id(Pb)].append(id(Pa))

    cliques = bron_kerbosch(G)
    for clique in cliques:
        clique_trees = [t for t in V if id(t) in clique]
        clique_T = grow_horizontally(*clique_trees, presort=True)
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
            t = Tree(
                o.start, r=o.length, p=o.period, children=[o.Index[0]], E=[0] + o.dE
            )
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

        while V or H:
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
