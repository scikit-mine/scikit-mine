"""
Periodic trees
"""
import array
import dataclasses
import datetime as dt
import re
import warnings
from collections import defaultdict
from functools import partial
from itertools import chain, combinations, count
from typing import List

import numpy as np
import pandas as pd
from sortedcontainers import SortedKeyList
from pyroaring import BitMap as Bitmap

from ..base import BaseMiner, InteractiveMiner
from ..utils import bron_kerbosch
from .cycles import PeriodicCycleMiner, extract_triples, merge_triples

L_PARENTHESIS = -np.log2(
    1 / 3
)  # length of either `(` or `)` when encoding tree patterns

shift_array = partial(array.array, "i")  # shifts are signed integers


def get_occs(node, E=None, tau=0, sort=True, r=None):
    """
    get occurences covered by a node (or tree)
    """  # TODO just check isinstance(Tree)
    if not E:
        E = shift_array([0] * (node._n_occs - 1))
    assert len(E) == node._n_occs - 1

    def _get_occs(node, acc, E, tau=0):
        dists = np.cumsum([0] + node.children_dists)
        inters = np.arange(tau, (node.r * node.p) + tau, node.p, dtype=np.int32)
        x_acc = 0
        y_acc = 0
        for inter in inters:
            for idy, dist, child in zip(count(), dists, node.children):
                if isinstance(child, Node):
                    new_tau = inter + dist  # + E[len(acc)]
                    _get_occs(child, acc, E, tau=new_tau)
                elif r is not None and len(acc) >= r:
                    return
                else:
                    # first one has no shift correction
                    e = E[len(acc) - 1] if acc else 0
                    if idy == 0:
                        x_acc += e
                        y_acc = 0
                    else:
                        y_acc += e
                    occ = (inter + dist + x_acc + y_acc, child)
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
    _leaves: list = dataclasses.field(init=False, repr=False)
    _nodes: list = dataclasses.field(init=False, repr=False)

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

        self._leaves = [c for c in self.children if not isinstance(c, Node)]
        self._nodes = [c for c in self.children if isinstance(c, Node)]
        # assert len(n_t) == 2

        self._size = 1 + len(self._leaves) + sum((c._size for c in self._nodes))
        self._n_occs = self.r * (
            len(self._leaves) + sum((c._n_occs) for c in self._nodes)
        )

    def mdl_cost_R(self, **event_frequencies):
        occs_min = -np.log2(min(map(event_frequencies.__getitem__, self._leaves)))
        return occs_min + sum((c.mdl_cost_R(**event_frequencies) for c in self._nodes))

    def mdl_cost_A(self, **event_frequencies):
        leaves_cost = list(map(event_frequencies.__getitem__, self._leaves))
        return (
            2 * L_PARENTHESIS
            + sum((c.mdl_cost_A(**event_frequencies) for c in self._nodes))
            - sum(map(np.log2, leaves_cost))
        )

    def mdl_cost_tau(self, dS):
        """
        Parameters
        ----------
        dS: int
            difference between first and last occurence from the original data
        """
        if self._nodes:
            first_node = self._nodes[0]
            maxv = dS - (first_node.r - 1) * first_node.p + 1
            return np.log2(maxv)
        return 0.0

    def mdl_cost_p0(self, dS, eps_oza=0):
        """compute mdl cost for p0"""
        if self._nodes:
            first_node = self._nodes[0]
            maxv = np.floor((dS - eps_oza) / (first_node.r - 1))
            return np.log2(maxv)
        return 0.0

    def mdl_cost_D(self, spanned, interleaving=False):
        """
        spanned is big_delta_star in page 16 and 17
        """
        # TODO : check we can do this kinf of arithmetic with datetimes
        if self._nodes:
            """ TODO What to to of the following lines ? see page 16/17
            first_node = self._nodes[0]
            delta_star_max = (
                (spanned - self.r + 1)
                if interleaving
                else min(self.p, np.floor(spanned / self.r))
            )
            """
        dists_cumsum = np.cumsum([0] + self.children_dists)
        res = np.log2(np.floor(spanned / (self.r - 1)))  # see bottom of page 16
        for idx, child in enumerate(self.children):
            if isinstance(child, Node):
                child_spanned = spanned - dists_cumsum[:idx]
                res += child.mdl_cost_D(child_spanned, interleaving=interleaving)
            else:
                res -= dists_cumsum[idx]  # FIXME this can give negative costs

        return res

    to_dict = dataclasses.asdict
    to_tuple = dataclasses.astuple

    def __len__(self):
        return self._size

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Node) and all(
            getattr(self, attr) == getattr(o, attr)
            for attr in ("r", "p", "children", "children_dists")
        )

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __str__(self):
        children_str = list(map(str, self.children))  # recursive call here
        dists_str = map(str, self.children_dists)
        event_str = [val for pair in zip(children_str, dists_str) for val in pair] + [
            children_str[-1]
        ]
        event_str = " - ".join(event_str)
        repeat = "{" + f"r={self.r}, p={self.p}" + "}"
        return f"{repeat} ({event_str})"

    @classmethod
    def from_str(cls, string, parse_dates=False):
        """
        Construct a node from a string of the form
        {r=5, p=7} ({r=3, p=2} (b - 2 - c) - 4 - a)
        """
        date_cons = dt.datetime if parse_dates else int

        r, p = re.search(r"\s*r=\s*(\d+),\s*p=\s*(\w+)", string).groups()
        r, p = int(r), date_cons(p)

        # TODO: use pos from previous extraction
        nodes_str = string[:-1].split("(", 1)[1]
        idx, dash_counter = 0, 0
        children, children_dists = list(), list()

        while idx < len(nodes_str):
            if nodes_str[idx] == "{":
                end = nodes_str[idx:].find(")") + 1
                child = cls.from_str(nodes_str[idx:end], parse_dates=parse_dates)
                children.append(child)
                idx = end
            if nodes_str[idx] == " ":
                idx += 1
                continue
            if nodes_str[idx] == "(":
                buf = nodes_str[idx]
                while nodes_str[idx] != ")":
                    idx += 1
                    buf += nodes_str[idx]
                node = cls.from_str(buf)
                children.append(node)
                dash_counter += node._n_occs
            if nodes_str[idx] == "-":
                dash_counter += 1
            else:
                next_item_match = re.search("\w+", nodes_str[idx:])
                next_item = next_item_match.group()
                if dash_counter % 2:
                    dist = date_cons(next_item)
                    children_dists.append(dist)
                else:
                    children.append(next_item)
                idx += next_item_match.end() - 1

            idx += 1

        return cls(r, p, children, children_dists)


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
     - tids
    """

    def __init__(self, tau, tids=None, E=None, *args, **kwargs):
        super(Tree, self).__init__(*args, **kwargs)
        self.tau = tau  # TODO : add tau in repr
        self.tids = tids or Bitmap()
        if E is None:
            self.E = shift_array([0] * (self._n_occs - 1))
            self.mdl_cost_E = 0
        else:
            assert hasattr(E, "__len__") and len(E) == self._n_occs - 1
            self.E = shift_array(E)
            self.mdl_cost_E = np.sum(np.abs(self.E))

        if self.tids:
            assert len(self.tids) == self._n_occs

    def get_occs(self, apply_shifts: bool = True):
        """
        unfold the tree and retrieve all occurences of the events it describes
        """
        return get_occs(self, tau=self.tau, E=self.E)

    def to_list(self):
        """Returns the tree as a list of (occurence, event) pairs"""
        return list(self.get_occs())

    def get_internal_nodes(self):
        """yield interal nodes"""
        return filter(lambda n: isinstance(n, Node), prefix_visitor(self))

    def to_node(self):
        """remove `tau` and `E` from the current object, return a new instance of Node"""
        return Node(
            self.r, self.p, children=self.children, children_dists=self.children_dists
        )

    def mdl_cost_D(self, S):
        """compute the cost of D given S
        # TODO rename S in D to avoid confusion

        compute the time spanned by this tree (the root node) and make recursive
        call on inner blocks (children)
        """
        if not self.tids:
            raise ValueError("we need tids to compute the time spanned by a block")
        spanned = S.index[self.tids[-1]] - S.index[self.tids[0]]
        return super().mdl_cost_D(spanned)

    def mdl_cost(self, D, dS, **event_frequencies):
        return (
            self.mdl_cost_A(**event_frequencies)
            + self.mdl_cost_R(**event_frequencies)
            + self.mdl_cost_p0(dS)
            + self.mdl_cost_tau(dS)
            + self.mdl_cost_E
            + self.mdl_cost_D(D)
        )

    def __str__(self):
        """Express this tree in a human-readable language"""
        return f"{self.tau} {super().__str__()}"

    def __repr__(self):
        repres = super(Tree, self).__repr__()
        pos = len(type(self).__name__) + 1
        return f"{repres[:pos]}tau={self.tau}, {repres[pos:]}"

    @classmethod
    def from_str(cls, string, parse_dates=False):
        tau_constructor = dt.datetime if parse_dates else int
        tau_match = re.search(r"\w+", string)
        node_str = string[tau_match.end() :]
        tau = tau_constructor(tau_match.group())
        node = Node.from_str(node_str)
        return cls(
            tau,
            r=node.r,
            p=node.p,
            children=node.children,
            children_dists=node.children_dists,
        )

    def __eq__(self, o):
        """Comparing trees, transactions ids are not checked, on purpose."""
        this_eq = isinstance(o, Tree) and self.tau == o.tau and self.E == o.E
        return this_eq and super().__eq__(o)

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)


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
    H = sorted(H, key=lambda e: e.tau)  # TODO : use Forest class and no need for this
    while H:  # for each distinc tree
        Tc = H[0]
        C = [t for t in H if Node.__eq__(t, Tc)]
        taus = np.array([_.tau for _ in C])
        cycles_tri = extract_triples(taus)  # TODO : pass `l_max`
        cycles_tri = merge_triples(cycles_tri)  # TODO: check if this step is mandatory

        for cycle_batch in cycles_tri:
            p_vect = np.floor(np.median(np.diff(cycle_batch, axis=1), axis=1)).astype(
                int  # TODO make call to median to just return an int
            )
            r = cycle_batch.shape[1]
            for idx, tau, p in zip(count(), cycle_batch[:, 0], p_vect):
                # create a new tree to make sure we don't mistankenly
                # manipulate references on the root
                sub_C = [_ for _ in C if _.tau in cycle_batch[idx]]
                tids = Bitmap.union(*(_.tids for _ in sub_C))
                taus_diffs = np.diff(taus) - p
                E = [
                    sub_C[idx].E.tolist() + [taus_diffs[idx]]
                    for idx in range(len(taus_diffs))
                ] + [sub_C[-1].E.tolist()]
                E = list(chain(*E))
                K = Tree(tau, r=r, p=p, children=[Tc.to_node()], tids=tids, E=E)
                # TODO : check cost (line 8 from algorithm 4)
                V_prime.append(K)
                H = [_ for _ in H if _ not in C]
        else:
            break

    return V_prime


def grow_horizontally(*trees, presort=True, S=None):
    """Grow trees horizontally"""
    if presort:
        trees = sorted(trees, key=lambda t: t.tau)
    p = trees[0].p
    # p = np.median([t.p for t in trees])
    r = min((_.r for _ in trees))
    children_dists = [b.tau - a.tau for a, b in zip(trees, trees[1:])]
    children = list(chain(*(_.children for _ in trees)))
    tau = trees[0].tau
    tids = Bitmap.union(*(_.tids for _ in trees))
    T = Tree(tau, r=r, p=p, children=children, children_dists=children_dists, tids=tids)
    E = list()

    trees = list(trees)

    if S is None:
        occs1 = sorted(chain(*(get_occs(t, tau=t.tau, E=t.E) for t in trees))) * r
        occs1 = occs1[1:]
    else:
        occs1 = S.iloc[tids[1:]].items()  # faster and safer
    occs2 = get_occs(T, tau=tau)[1:]  # cover instances with a perfect tree
    # E = shift_array([v for tup in zip(*[t.E for t in trees]) for v in tup])
    E = shift_array([a[0] - b[0] for a, b in zip(occs1, occs2)])  # FIXME page 21
    assert len(E) == T._n_occs - 1
    T.E = E
    return T


def combine_horizontally(V: list, S=None):
    """Grow trees vertically"""
    H_prime = list()
    G = defaultdict(list)
    C = [(Pa, Pb) for Pa, Pb in combinations(V, 2) if Pb.tau <= Pa.tau + Pa.p]

    for Pa, Pb in C:
        K = grow_horizontally(Pa, Pb, S=S)
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


def greedy_cover(candidates: List[Tree], D, dS, k=10, **event_frequencies):
    """
    Greedy cover, as described at the top of page 30 in the original paper

    the set P of selected patterns is empty. Let O be the set of event
    occurrences covered so far, also initially empty.

    In each roud; the pattern p with smallest value of L(P) / |occs(P) \ O|
    among remaining candidates is selected.

    p is added to P, O is updated and the selection proceeds to the next round,
    until the `k` round is reached (at most).
    """
    big_P = list()
    big_O = Bitmap()  # set of event occurences covered so far
    candidates = candidates.copy()

    for _ in range(k):
        p = min(
            candidates,
            key=lambda p: p.mdl_cost(D, dS, **event_frequencies)
            / (len(p.tids - big_O) or 0.001),
        )
        big_O |= p.tids
        big_P.append(p)
        candidates.remove(p)

        if not candidates:
            break

    return big_P


class PeriodicPatternMiner(BaseMiner, InteractiveMiner):
    """
    Mining Periodic Pattern with a MDL criterion

    Warning: this implementation is unstable.

    Implementation of periodic tree mining.

    This first extract cycles from the input data, and then combine these cycles
    into more complex tree structures.

    A tree is defined as a 3-tuple of the form
    :math: `\tau`, `C`, `E`


    Parameters
    ----------
    k: int, default=20
        Number of trees to keep from the set of candidates.
        Trees will be evaluated based on their

    See Also
    --------
    skmine.periodic.PeriodicCycleMiner
    """

    def __init__(self, k=20, **cycle_miner_kwargs):
        # TODO : pass instance of PeriodicCycleMiner, check is_fitted
        self.cycle_miner = PeriodicCycleMiner(**cycle_miner_kwargs)
        self.codetable = list()  # list of (tree, cost) pairs
        self.event_frequencies = defaultdict(int)  # throw 1 by default
        self.k = k
        self.dS = 1

    def prefit(self, S):
        cycles = self.cycle_miner.fit_discover(S, shifts=True, tids=True)
        singletons = list()
        for o in cycles.itertuples():
            t = Tree(
                o.start,
                r=o.length,
                p=o.period,
                children=[o.Index[0]],
                E=o.dE,
                tids=o.tids,
            )
            singletons.append(t)

        singletons = sorted(singletons, key=lambda t: t.tau)

        self.event_frequencies = (S.value_counts() / len(S)).to_dict()
        self.dS = S.index.max() - S.index.min()
        return singletons

    def generate_candidates(self, singletons, S=None):
        """
        Generate candidate trees be successive vertical/horizontal combinations

        Starting from singleton trees (~ cycles)

        See Also
        --------
        prefit
        """
        H = singletons  # list of horizontal combinations
        V = singletons  # list of vertical combinations

        C = list()

        while V or H:
            V_prime = combine_vertically(H)
            H_prime = combine_horizontally(V, S=S)
            V = V_prime
            H = H_prime
            C += H + V  # TODO yield ?

        return C

    def evaluate(self, trees, D):
        """
        Evaluate candidate trees

        See the "greedy cover" method described at the top of page 30 from
        the original paper
        """
        # TODO keep event_frequencies as an attribute
        # TODO avoid calling mdl_cost here, this is already done in greedy_cover
        dS = D.index.max() - D.index.min()
        trees = greedy_cover(trees, D, dS, k=self.k, **self.event_frequencies)
        costs = [t.mdl_cost(D, dS, **self.event_frequencies) for t in trees]
        return list(zip(trees, costs))

    def update(self, t: Tree):
        self.codetable.append(
            (t, 0)
        )  # FIXME cost of 0 here, just for interactive mining, but to be fixed
        if len(self.codetable) > self.k:
            warnings.warn("current number of trees exceeds the `k` parameter")

    def fit(self, D):
        """
        Discover periodic patterns (in the form of trees) from a sequence of event `D`

        This iteratively refines the set of trees by successive vertical/horizontal
        combinations, starting from single-node trees describing `cycles`.

        The resulting model is a list of periodic trees.
        """
        singletons = self.prefit(D)
        C = self.generate_candidates(singletons, D)
        self.codetable = self.evaluate(C, D)
        return self

    def discover(self):
        """
        Overview on the discovered trees
        """
        data = [dict(description=str(t[0]), cost=t[1]) for t in self.codetable]
        return pd.DataFrame(data)
