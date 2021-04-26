# TODO : visitor
import copy
from itertools import zip_longest

import numpy as np

from .cycles import PeriodicCycleMiner, extract_triples, merge_triples


def get_occs(node, tau=0):
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
        self.children_dists = children_dists
        self.children = children

    def size(self):
        return sum((1 for _ in prefix_visitor(self)))

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


def combine_vertically(H: list):
    """
    combine trees verically, by detecting cycles on their `tau`s
    """
    V_prime = list()
    while H:  # for each distinc tree
        Tc = H[0]
        C = [t for t in H if (t.r == Tc.r and t.p == Tc.p)]  # TODO Tc == t
        taus = np.array([_.tau for _ in C])
        cycles_tri = extract_triples(taus)  # TODO : pass `l_max`
        cycles_tri = merge_triples(cycles_tri)
        for cycle_batch in cycles_tri:
            p_vect = np.median(np.diff(cycle_batch, axis=1), axis=1)
            r = cycle_batch.shape[1]
            for tau, p in zip(cycle_batch[:, 0], p_vect):
                # create a new tree to make sure we don't mistankenly
                # manipulate references on the root
                K = Tree(tau, r=r, p=p, children=[Tc])
                # TODO : check cost (line 8 from algorithm 4)
                V_prime.append(K)
                H = [
                    _ for _ in H if _ not in C
                ]  # FIXME : this differs from the original paper
        else:
            break

    return V_prime


class PeriodicPatternMiner:
    def __init__(self, max_length=100):
        # TODO : pass instance of PeriodicCycleMiner, check is_fitted
        self.cycle_miner = PeriodicCycleMiner(max_length=max_length)

    def _prefit(self, D):
        cycles = self.cycle_miner.fit_discover(D)
        singletons = list()
        for o in cycles.itertuples():
            t = Tree(o.start, r=o.length, p=o.period, children=[o.Index[0]])
            singletons.append(t)

        return singletons

    def fit(self, D):
        singletons = self._prefit(D)
        # singletons = [Tree(tau=)]
        H = copy.deepcopy(singletons)  # list of horizontal combinations
        V = copy.deepcopy(singletons)  # list of vertical combinations

        while V:  # TODO while H or V
            V_prime = combine_vertically(H)
            H_prime = H  # TODO combine_horizontally(V, P)
            V = V_prime
            H = H_prime

        # TODO P = greedy_cover(C, S), return P
        return H
