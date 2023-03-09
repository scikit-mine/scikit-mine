import copy
import pdb
import re

import numpy as np

from .class_patterns import l_to_key, key_to_l, OPT_TO


def getEDict(oStar, E=[]):
    """
    Construct a dictionary that maps each unique identifier in the input list of occurrences from `oStar` to
    it's corresponding value in the list of errors `E`.

    Parameters
    ----------
    oStar : list
        A list of tuples representing occurences in the tree structure composed of
        three items (t0, event, position in tree).
    E : list, default=[]
        A list of values of shift corrections (errors) that correspond to the unique identifiers in `oStar`.
        If `len(E)` is less than `len(oStar) - 1`, the remaining values are set to zero.

    Returns
    -------
    dict
        A dictionary that maps each unique identifier in `oStar` to a corresponding value in `E`.
    """
    if len(E) >= len(oStar) - 1:
        Ex = [0] + list(E[:len(oStar) - 1])
    else:
        Ex = [0 for o in oStar]
    oids = [o[-1] for o in oStar]
    return dict(zip(*[oids, Ex]))


def getEforOccs(map_occs, occs):
    """
    FIXME : to be explained

    Constructs the list of errors
    # TODO : WARNING! WRONG, this is using absolute errors...

    Parameters
    ----------
    map_occs
    occs

    Returns
    -------

    """
    return [t - map_occs.get(oid, t) for (t, alpha, oid) in occs]


def codeLengthE(E):
    """
    L(E) = 2 |E| + ∑_{e∈E} |e|

    Parameters
    ----------
    E : list
        The list of errors

    Returns
    -------
        int
    L(E)
    """
    return np.sum([2 + np.abs(e) for e in E])


class Pattern(object):
    """
    This class models the patterns from a tree structure.

    Attributes:
    -----------
    transmit_Tmax : bool, default=True
        Whether to transmit Tmax.
    allow_interleaving : bool, default=True
        Whether to allow interleaving.
    next_id : int, default=1
        ID of the next node to be added to the pattern.
    nodes : dict
        Dictionary representing the pattern from a tree structure, where keys are node IDs
        and values are dictionaries with the following keys:
            - parent: ID of the parent node. None if node is the root.
            - children: List of tuples representing the children of the node.
                Each tuple has two elements:
                    - ID of the child node
                    - distance between the parent node and the child node
            - event: The event associated with the node. None if node is an inner node.
            - p: The period of the node. None if node is not a root node.
            - r: The repetition of the node. None if node is not a root node.
    """
    transmit_Tmax = True
    allow_interleaving = True

    def __init__(self, event=None, r=None, p=None):
        self.next_id = 1
        self.nodes = {}
        if event is not None:
            if type(event) is dict:  # actually a tree
                self.next_id = max(list(event.keys())) + 1
                self.nodes = event
            else:  # a simple event cycle
                self.nodes[self.next_id] = {"parent": 0, "event": event}
                self.nodes[0] = {"parent": None, "p": p,
                                 "r": r, "children": [(self.next_id, 0)]}
                self.next_id += 1
        else:
            self.nodes[0] = {"parent": None, "children": []}

    def copy(self):
        """
        Deep copy of a pattern

        Returns
        -------
            None
        """
        pc = Pattern()
        pc.next_id = self.next_id
        pc.nodes = copy.deepcopy(self.nodes)
        return pc

    def mapEvents(self, map_evts):
        """
        Mapping of event indexes to their real textual names. Pattern modification in place.

        Parameters
        ----------
        map_evts : a list where the value at index i corresponds to the name of the event i

        Returns
        -------
            None
        """
        for nid in self.nodes.keys():
            if self.isLeaf(nid):
                self.nodes[nid]["event"] = map_evts[self.nodes[nid]["event"]]

    def getTranslatedNodes(self, offset):
        """
        Translate the ids of nodes by a certain offset

        Parameters
        ----------
        offset : int
            Number indicating how much the node/leaf ids are shifted

        Returns
        -------
        dict
            The translated nodes
        int
            The next available node id after the translation
        dict
            A dictionary mapping the original node ids to the translated node ids
        """
        nodes = {}
        map_nids = dict([(kk, offset + ki)
                         for ki, kk in enumerate(self.nodes.keys())])
        nks = list(self.nodes.keys())
        while len(nks) > 0:
            fn = nks.pop()
            tn = map_nids.get(fn, fn)
            tmp = {}
            for k, v in self.nodes[fn].items():
                if k == "parent":
                    tmp[k] = map_nids.get(v, v)
                elif k == "children":
                    tmp[k] = [(map_nids.get(c[0], c[0]), c[1]) for c in v]
                else:
                    tmp[k] = v
            nodes[tn] = tmp
        return nodes, offset + len(map_nids), map_nids

    def merge(self, patt, d, anchor=0):
        """
        Merge a given pattern into the current pattern instance.

        Parameters
        ----------
        patt : Pattern
            The pattern to merge

        d : int
            time distance between the two patterns

        anchor : int, default=0
            Node id from which the merge is done

        Returns
        -------
        dict
            A dictionary having for key the initial ids of patt and for values the new ids after merging with
            the Pattern instance.
        """
        if not self.isInterm(anchor):
            anchor = 0
        nodes, self.next_id, map_nids = patt.getTranslatedNodes(self.next_id)
        nodes[map_nids[0]]["parent"] = anchor
        self.nodes[anchor]["children"].append((map_nids[0], d))
        self.nodes.update(nodes)
        return map_nids

    def append(self, event, d, anchor=0):
        """
        Append an event with a specific distance and a possible anchor node id to the pattern.

        Parameters
        ----------
        event : int
            The event to be added
        d : int
            Time distance from the anchor node
        anchor : int
            Node id from which the addition is made

        Returns
        -------
            None
        """
        if not self.isInterm(anchor):
            anchor = 0
        self.nodes[self.next_id] = {"parent": anchor, "event": event}
        self.nodes[anchor]["children"].append((self.next_id, d))
        self.next_id += 1

    def repeat(self, r, p):
        """
        Repeat a pattern r times with a period p.

        Parameters
        ----------
        r : int
            Number of repetitions
        p : int
            Period between occurrences

        Returns
        -------
            None
        """
        if "r" not in self.nodes[0]:
            self.nodes[0]["p"] = p
            self.nodes[0]["r"] = r
            return
        self.nodes[0]["parent"] = 0
        self.nodes[self.next_id] = self.nodes.pop(0)
        if "children" in self.nodes[self.next_id]:
            for (nc, _) in self.nodes[self.next_id]["children"]:
                self.nodes[nc]["parent"] = self.next_id
        self.nodes[0] = {"parent": None, "p": p,
                         "r": r, "children": [(self.next_id, 0)]}
        self.next_id += 1

    def isNode(self, nid):
        """
        Check if node id (nid) exists in the pattern.

        Parameters
        ----------
        nid : int
            Node id to check

        Returns
        -------
        bool
            True if the node id exists in the pattern, False otherwise.
        """
        return nid in self.nodes

    def isInterm(self, nid):
        """
        Check if node id (nid) is an intermediate node. It is not a leaf.

        Parameters
        ----------
        nid : int
            Node id to check

        Returns
        -------
        bool
            True if the node id is an intermediate node, False otherwise.
        """
        return self.isNode(nid) and "children" in self.nodes[nid]

    def isLeaf(self, nid):
        """
        Check if node id (nid) is a leaf.

        Parameters
        ----------
        nid : int
            Node id to check

        Returns
        -------
            True if the node id is a leaf, False otherwise.
        """
        return self.isNode(nid) and "children" not in self.nodes[nid]

    def getNidsRightmostLeaves(self, nid=0, rightmost=True):
        """
        Get the rightmost leaves ids under the node with id `nid`.

        Parameters
        ----------
        nid : int, default=0
            The id of the node to start the computation

        rightmost : bool, default=True
            Wheter the current node is the rightmost child of its parent

        Returns
        -------
        list
            A list of ids of the rightmost leaves.
        """
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            rightmost_nids = []
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                rightmost_nids.extend(self.getNidsRightmostLeaves(
                    nn[0], ni == len(self.nodes[nid]["children"]) - 1))
            return rightmost_nids
        else:
            if rightmost:
                return [nid]
            return []

    def getOccsStar(self, nid=0, pref=[], time=0):
        """
        Get the list of timestamp-event pairs from the pattern reconstructed from his tree before correction.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done
        pref : list, default=[]
        time : int, default=0

        Returns
        -------
        List[Tuple[int, int, str]]
            List of timestamp-event pairs (timestamp, event, a string representing the position in the tree)
            FIXME : explain more the latest item
        """
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            occs = []
            for i in range(self.nodes[nid]["r"]):
                tt = time + i * self.nodes[nid]["p"]
                for ni, nn in enumerate(self.nodes[nid]["children"]):
                    tt += nn[1]
                    occs.extend(self.getOccsStar(nn[0], [(ni, i)] + pref, tt))
            return occs
        else:
            return [(time, self.nodes[nid]["event"], l_to_key(pref[::-1]))]

    def getTimesNidsRefs(self, nid=0, pref=[], time=0):
        """
        The only difference with the `getOccsStar` method is that the second element of the tuple is the node id and
        not the event.
        """
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            occs = []
            for i in range(self.nodes[nid]["r"]):
                tt = time + i * self.nodes[nid]["p"]
                for ni, nn in enumerate(self.nodes[nid]["children"]):
                    tt += nn[1]
                    occs.extend(self.getTimesNidsRefs(nn[0], [(ni, i)] + pref, tt))
            return occs
        else:
            return [(time, nid, l_to_key(pref[::-1]))]

    def getCCorr(self, k, Ed):
        """
        Adds shift corrections to calculated theoretical timestamps

        Parameters
        ----------
        k : int
            Indicate how far we want the sum of the errors to go
        Ed : dict
            For each occurrence ('0,0' for example) indicate the shift correction associated

        Returns
        -------
        int
            Return the sum of the errors up to k
        """
        return np.sum([Ed[k]] + [Ed[kk] for kk in self.gatherCorrKeys(k)])

    def getOccs(self, oStar, t0, E=[]):
        """
        Get the list of timestamp from the pattern reconstructed from his tree after correction.

        Parameters
        ----------
        oStar : list
            A list of tuples representing occurences in the tree structure composed of
            three items (t0, event, position in tree).

        t0 : int
            Start time of the sequence

        E : dict
            A dictionary associating to each occurrence its error
            Example : {'0,0': 5}

        Returns
        -------
        list
            List of timestamp from the pattern after correction
        """
        if type(E) is dict:
            Ed = E
        else:
            Ed = getEDict(oStar, E)
        return [o[0] + t0 + self.getCCorr(o[-1], Ed) for o in oStar]

    def getCovSeq(self, t0, E=[]):
        """
        Similar as getOccs but returned list of tuples where the first item correspond to the timestamp after correction
        and the second item is the event.

        Parameters
        ----------
        t0 : int
            Start time of the sequence

        E : dict
            A dictionary associating to each occurrence its error
            Example : {'0,0': 5}

        Returns
        -------
        list[Tuples]
            First item of each tuple correspond to the timestamp after correction and the second item is the event.
        """
        oStar = self.getOccsStar()
        # all last elements in the previous tuple associated to his shift correction
        Ed = getEDict(oStar, E)
        return [(o[0] + t0 + self.getCCorr(o[-1], Ed), o[1]) for o in oStar]

    def getNidFromKey(self, k):
        """
        FIXME : to be explained

        Parameters
        ----------
        k

        Returns
        -------

        """
        if len(k) == 0:
            return 0
        if type(k) is list:
            key_ints = copy.deepcopy(k)
        else:
            key_ints = key_to_l(k)
        if len(key_ints) == 0:
            return None  # something went wrong
        current_node, level = (0, 0)
        while len(key_ints) > level > -1:
            if self.isInterm(current_node) and key_ints[level][0] < len(self.nodes[current_node]["children"]):
                if key_ints[level][0] == -1:
                    if level + 1 == len(key_ints):
                        return current_node
                current_node = self.nodes[current_node]["children"][key_ints[level][0]][0]
                level += 1
            else:
                level = -1
        if level >= 0:
            return current_node
        return None

    def getKeyFromNid(self, nid=0, rep=0):
        """
        FIXME : to be explained

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        rep : int, default=0
            If -1, the number of repetitions associated with the node on the left is automatically calculated,
             otherwise, it is set to the value passed in parameter.

        Returns
        -------
        str
        """
        tt = self.getKeyLFromNid(nid, rep)
        if tt is not None and self.isInterm(nid):
            if rep == -1:
                tt.append((-1, self.nodes[nid]["r"] - 1))
            else:
                tt.append((-1, rep))

        if tt is not None:
            return l_to_key(tt)

    def getKeyLFromNid(self, nid=0, rep=0):
        """
        FIXME : to be explained
        Get the id node to the left of the nid indicated in parameter with the number of associated repetitions - 1
        of the parent node.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        rep : int, default=0
            If -1, the number of repetitions associated with the node on the left is automatically calculated,
             otherwise, it is set to the value passed in parameter.

        Returns
        -------
        list[Tuple]
            Returns a list where each element is a tuple containing 2 items.
            The first one corresponds to the node id of the leaf on the left and
            the second to the number of repetitions - 1 of the parent or rep passed in parameter.

            Furthermore, if there is no leaf to the left or directly at its level then the function
            also returns a tuple (0, rep from the root)
        """
        if not self.isNode(nid):
            return None
        parent = self.nodes[nid]["parent"]
        if parent is None:
            return []
        else:
            cid = 0
            while len(self.nodes[parent]["children"]) > cid > -1:
                if self.nodes[parent]["children"][cid][0] == nid:
                    kl = self.getKeyLFromNid(parent, rep)
                    if kl is None:
                        return None
                    else:
                        if rep == -1:
                            return kl + [(cid, self.nodes[parent]["r"] - 1)]
                        else:
                            return kl + [(cid, rep)]
                else:
                    cid += 1
        return None

    def getLeafKeyFromKey(self, k, cid=0, rep=0):
        """
        FIXME : to be explained

        Parameters
        ----------
        k
        cid
        rep

        Returns
        -------

        """
        if type(k) is list:
            key_ints = copy.deepcopy(k)
        else:
            key_ints = key_to_l(k)
        nid = self.getNidFromKey(k)
        suff = []
        first_rep = None
        if self.isInterm(nid):
            if len(key_ints) > 0 and key_ints[-1][0] == -1:
                _, first_rep = key_ints.pop()
            while self.isInterm(nid):
                ccid = cid
                if ccid == -1:
                    ccid = len(self.nodes[nid]["children"]) - 1

                rrep = rep
                if first_rep is not None:
                    rrep = first_rep
                    first_rep = None
                elif rrep < 0:
                    rrep = self.nodes[nid]["r"] - 1
                suff.append((ccid, rrep))
                nid = self.nodes[nid]["children"][ccid][0]

        if nid is not None:
            return l_to_key(key_ints + suff)
        return None

    def gatherCorrKeys(self, k):
        """
        FIXME : to be explained

        Parameters
        ----------
        k

        Returns
        -------

        """
        if type(k) is list:
            key_ints = copy.deepcopy(k)
        else:
            key_ints = key_to_l(k)

        cks = []
        if len(key_ints) == 1 and key_ints[0][0] == -1:
            for i in range(key_ints[0][1]):
                cks.append(self.getLeafKeyFromKey([(-1, i)]))
        elif len(key_ints) >= 1:
            last = key_ints.pop()
            for i in range(last[0]):
                cks.append(self.getLeafKeyFromKey(key_ints + [(i, last[1])]))
            for i in range(last[1]):
                cks.append(self.getLeafKeyFromKey(key_ints + [(-1, i)]))
            cks.extend(self.gatherCorrKeys(key_ints))
        return cks

    def getEventsList(self, nid=0, add_delimiter=True):
        """
        Get the list of events from a node id and with or without delimiters (parenthesis for child events).
        This list does not include repeats.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done
        add_delimiter : bool, default=True
            Adding a delimiter (parenthesis) to differentiate child events in the tree

        Returns
        -------
        list
            A list of tree events.
            Example with:
            - `add_delimiter=True` : ['(', '4', '7', '(', '8', ')', ')']
            - `add_delimiter=True` : ['4', '7', '8']
        """
        if not self.isNode(nid):
            return ""  # was initially openB_str + closeB_str` but these two variables do not exist
        if self.isInterm(nid):
            ll = []
            for nn in self.nodes[nid]["children"]:
                ll.extend(self.getEventsList(nn[0], add_delimiter))
            if add_delimiter:
                return ["("] + ll + [")"]
            else:
                return ll
        else:
            return ["%s" % self.nodes[nid]["event"]]

    def getEventsMinor(self, nid=0, rep=False):
        """
        Get the list of events under a node.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done
        rep : bool, default=False
            If true, the repetitions of the child events (leaves) of the nid node are calculated
            otherwise only the repetitions of the child nodes are calculated.

        Returns
        -------
        list
            The list of events associated to nid
        """
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            ll = []
            for nn in self.nodes[nid]["children"]:
                ll.extend(self.getEventsMinor(nn[0], True))
            if rep:
                return self.nodes[nid]["r"] * ll
            else:
                return ll
        else:
            return [self.nodes[nid]["event"]]

    def getTreeStr(self, nid=0, level=0, map_ev=None):
        """
        Generate the visualization of the tree representing the pattern

        Parameters
        ----------
        nid : int, default=0
            Node id from which the display is made

        level : int, default=0
            Indentation of the tree. The higher this number, the more indented the leaves/nodes are.

        map_ev : dict
            Associate to each event id, it's event name

        Returns
        -------
            str
        """
        if not self.isNode(nid):
            return ("\t" * level) + "()\n"

        if self.isInterm(nid):
            ss = "%s|_ [%s] r=%s p=%s\n" % (
                ("\t" * level), nid, self.nodes[nid]["r"], self.nodes[nid]["p"])
            for nn in self.nodes[nid]["children"]:
                ss += "%s| d=%s\n" % (("\t" * (level + 1)), nn[1])
                ss += self.getTreeStr(nn[0], level + 1)
            return ss
        else:
            if map_ev is not None:
                return "%s|_ [%s] %s\n" % (
                    ("\t" * level), nid, map_ev.get(self.nodes[nid]["event"], self.nodes[nid]["event"]))
            else:
                return "%s|_ [%s] %s\n" % (("\t" * level), nid, self.nodes[nid]["event"])

    def __str__(self, nid=0, map_ev=None, leaves_first=False):
        """
        Display the pattern as a string. The tree representing the pattern is traversed in depth from left to right.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the display is made
        map_ev : dict
            Associate to each event id, it's event name
        leaves_first : bool, default=False
            If True, leaves are displayed first, otherwise last.

        Returns
        -------
        str
            A string representing the pattern with the form
        """
        if not self.isNode(nid):
            return ""
        if self.isInterm(nid):
            ss = "[r=%s p=%s]" % (self.nodes[nid].get(
                "r", "-"), self.nodes[nid].get("p", "-"))
            sc = ""
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                if ni > 0:
                    sc += " [d=%s] " % nn[1]
                sc += self.__str__(nn[0], map_ev, leaves_first)
            if leaves_first:
                return "(" + sc + ")" + ss
            return ss + "(" + sc + ")"
        else:
            if map_ev is not None:
                return "%s" % map_ev.get(self.nodes[nid]["event"], self.nodes[nid]["event"])
            else:
                return "%s" % self.nodes[nid]["event"]

    def pattKey(self, nid=0):
        """
        Displays the tree/pattern as a string

        Parameters
        ----------
        nid : int, default=0
            Node id from which the string is calculated

        Returns
        -------
        str
            A string of form
            "[r,p](left_child_event_id-time_distance_left_child_to_just_to_right_child-just_to_right_child_event_id-..."
        """
        if not self.isNode(nid):
            return ""
        if self.isInterm(nid):
            ss = "[%s,%s]" % (self.nodes[nid]["r"], self.nodes[nid]["p"])
            sc = ""
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                if ni > 0:
                    sc += "-%s-" % nn[1]
                sc += self.pattKey(nn[0])
            return ss + "(" + sc + ")"
        else:
            return "%s" % self.nodes[nid]["event"]

    def pattMinorKey(self, nid=0):
        """
        Displays the tree/pattern as a string except the first level where it starts (nid)

        Parameters
        ----------
        nid : int, default=0
            Node id from which the string is calculated

        Returns
        -------
        str
            A string of form
            "left_event-[time_distance_left_child_to_just_to_right_child]-just_to_right_event-[r,p](event...)..."

        """
        if not self.isNode(nid):
            return ""
        if self.isInterm(nid):
            sc = ""
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                if ni > 0:
                    sc += "-[%s]-" % nn[1]
                sc += self.pattKey(nn[0])
            return sc
        else:
            return ""

    def pattMajorKey(self, nid=0):
        """
        Get the repetition number and the period of a node id (nid) as a string

        Parameters
        ----------
        nid : int, default=0
            Node id from which the string is calculated

        Returns
        -------
        str
            A string of form : "[r,p]" of nid
        """
        if not self.isNode(nid):
            return ""
        if self.isInterm(nid):
            return "[%s,%s]" % (self.nodes[nid]["r"], self.nodes[nid]["p"])
        else:
            return "[]"

    def pattMajorKey_list(self, nid=0):
        """
        Get the repetition number and the period of a node id (nid) as a list

        Parameters
        ----------
        nid : int, default=0
            Node id from which the list is calculated

        Returns
        -------
        list
            A list of form : [r,p] of nid
        """
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            return [self.nodes[nid]["r"], self.nodes[nid]["p"]]
        else:
            return []

    def nodeP(self, nid=0):
        """
        Get the period of a certain node id

        Parameters
        ----------
        nid : int, default=0
            Node id from which we want the period

        Returns
        -------
        int
            Returns the period of the node if it is a node of the tree and not a leaf, otherwise returns 0.
        """
        if not self.isNode(nid) or not self.isInterm(nid):
            return 0
        else:
            return self.nodes[nid]["p"]

    def nodeR(self, nid=0):
        """
        Get the repetition number of a certain node id

        Parameters
        ----------
        nid : int, default=0
            Node id from which we want the repetition number

        Returns
        -------
        int
            Returns the repetition number of the node if it is a node of the tree and not a leaf, otherwise returns 0.
        """
        if not self.isNode(nid) or not self.isInterm(nid):
            return 0
        else:
            return self.nodes[nid]["r"]

    def getMajorOccs(self, occs):
        """
        Get the main timestamps from occs and the tree structure

        Only used when complex pattern are desired.

        Parameters
        ----------
        occs : list
            The list of timestamps/occurrences  of the events of the pattern

        Returns
        -------
        list
            The main timestamps of the pattern according to the root repetition number.
        """
        if self.getDepth() > 1 or self.getWidth() > 1:
            r = self.nodeR(0)
            len_ext_blck = len(occs) // r
            return occs[::len_ext_blck]
        return occs

    def factorizeTree(self, nid=0):
        """
        Factor the pattern tree if it can.
        This can be done if the children of a node have only one child and if these nodes have the same period
        and the same repetition.

        When 2 nodes are merged, the id of the deleted node is lost and the following ids are not shifted.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            None
        """
        ch = self.nodes[nid]["children"]
        anchor = ch[0][0]
        nch = [(self.nodes[nn[0]]["children"][0][0], nn[1]) for nn in ch]
        for nn in nch:
            self.nodes[nn[0]]["parent"] = anchor
        self.nodes[anchor]["children"] = nch
        for nn in ch[1:]:
            del self.nodes[nn[0]]
        self.nodes[nid]["children"] = [(anchor, 0)]

    def canFactorize(self, nid=0):
        """
        Indicates whether the pattern/tree can be factored or not.
        This can be done if the children of a node have only one child and if these nodes have the same period
        and the same repetition.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            list of all nodes of ids where the tree can be factored
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            f = []
            for nn in self.nodes[nid]["children"]:
                f.extend(self.canFactorize(nn[0]))
            if len(self.nodes[nid]["children"]) > 1:
                # intermediate nodes with single child
                if all([len(self.nodes[nn[0]].get("children", [])) == 1 for nn in self.nodes[nid]["children"]]):
                    # same length and period
                    if len(set([(self.nodes[nn[0]]["p"], self.nodes[nn[0]]["r"]) for nn in
                                self.nodes[nid]["children"]])) == 1:
                        f.append(nid)
            return f
        else:
            return []

    def getCyclePs(self, nid=0):
        """
        Get all p (period) values in the tree

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            list of all p values
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            rs = [self.nodes[nid]["p"]]
            for nn in self.nodes[nid]["children"]:
                rs.extend(self.getCyclePs(nn[0]))
            return rs
        else:
            return []

    def getCycleRs(self, nid=0):
        """
        Get all r (repetition) values in the tree

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            list of all r values
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            rs = [self.nodes[nid]["r"]]
            for nn in self.nodes[nid]["children"]:
                rs.extend(self.getCycleRs(nn[0]))
            return rs
        else:
            return []

    def getNbLeaves(self, nid=0):
        """
        Count the number of leaves in the tree. It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            int
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return np.sum([self.getNbLeaves(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getNbOccs(self, nid=0):
        """
        From the tree, count the total number of occurrences/timestamps.
        It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            int
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return self.nodes[nid]["r"] * np.sum([self.getNbOccs(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getDepth(self, nid=0):
        """
        Get the depth of the tree. It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            int
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return 1 + np.max([self.getDepth(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 0

    def getWidth(self, nid=0):
        """
        Get the width of the tree. It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            int
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return np.sum([self.getWidth(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getAlphabet(self, nid=0):
        """
        Get all id events from the tree. It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            set
        """
        # recursively collects all the different events
        if not self.isNode(nid):
            return set()
        if self.isInterm(nid):
            return set().union(*[self.getAlphabet(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return {self.nodes[nid]["event"]}

    def isSimpleCycle(self, nid=0):
        """
        Checks whether a tree is a simple cycle or not. It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            bool
        """
        return self.getDepth(nid) == 1 and self.getWidth(nid) == 1

    def isNested(self, nid=0):
        """
        Checks whether a tree is nested (depth > 1 and width == 1) or not.
        It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            bool
        """
        return self.getDepth(nid) > 1 and self.getWidth(nid) == 1

    def isConcat(self, nid=0):
        """
        Checks whether a tree is concat (depth == 1 and width > 1) or not.
        It also works from a certain node id of the tree.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            bool
        """
        return self.getDepth(nid) == 1 and self.getWidth(nid) > 1

    def getTypeStr(self):
        """
        Get the type of pattern (simple, nested, concat or other)

        Returns
        -------
            str
        The type of the pattern
        """
        if self.isSimpleCycle():
            return "simple"
        elif self.isNested():
            return "nested"
        elif self.isConcat():
            return "concat"
        else:
            return "other"

    def codeLengthEvents(self, adjFreqs, nid=0):
        """
        Compute the code length from a node id
        L(α) = − log(fr (α)) = − log(∣ ∣S(α)∣ ∣ / |S|)

        """
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            return np.sum([-np.log2(adjFreqs["("]), -np.log2(adjFreqs[")"])] +
                          [self.codeLengthEvents(adjFreqs, nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return -np.log2(adjFreqs[self.nodes[nid]["event"]])

    def getMinOccs(self, nbOccs, min_occs, nid=0):
        """
        Recursively collects info on the least occurring event in each block to be used to determine
        the code length for r_X. Each intermediate node B_X is associated with the period p_X and length r_X of the
        corresponding cycle.

        For a block B_X, the number of repetitions of the block cannot be larger than the number of occurences of the
        least frequent event participating in the block, denoted as p(B_X). We can thus encode the sequence of cycle
        lengths R with code of length L(R) = sum(L(r_X)) = sum(log(p(B_X)))

        Parameters
        ----------
        nbOccs : dict
            Associates for each event the number of times it appears in the dataset

        min_occs : list
            The list to be returned

        nid : int, default=0
            Node id from which the search is launched

        Returns
        -------
        list
            The min_occs list containing for each node of the tree, the minimum number of times an event participates
            in this block.
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            min_r = np.min([self.getMinOccs(nbOccs, min_occs, nn[0])
                            for nn in self.nodes[nid]["children"]])
            min_occs.append(min_r)
            return min_r
        else:
            return nbOccs[self.nodes[nid]["event"]]

    def getRVals(self, nid=0):
        """
        Recursively collects info on the least occurring event in each block to be used to determine the code length
        for r_X.

        Parameters
        ----------
        nid : int, default=0
            Node id from which the search is launched

        Returns
        -------
        list
            Get all R values from a certain node id
        """
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            rs = [self.nodes[nid]["r"]]
            for nn in self.nodes[nid]["children"]:
                rs.extend(self.getRVals(nn[0]))
            return rs
        else:
            return []

    def codeLengthR(self, nbOccs, nid=0):
        """
        Determine the code length for r_X based on info on the least occurring event in each block.
        L(R) = sum(L(r_X)) = sum(log(p(B_X)))

        Parameters
        ----------
        nbOccs : list
            Associates for each event the number of times it appears in the dataset
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            float
        The encoded length of the sequence of cycle lengths R
        """
        min_occs = []
        self.getMinOccs(nbOccs, min_occs, nid)
        rs = self.getRVals()
        clrs = np.log2(min_occs)
        return np.sum(clrs)

    def cardO(self, nid=0):
        """
        Computes the number of occurrences generated by a pattern

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            int
        The number of occurences generated by a pattern from a specific node id
        """
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            return self.nodes[nid]["r"] * np.sum([self.cardO(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getE(self, map_occs, nid=0):
        return getEforOccs(map_occs, self.getOccsStar(nid, time=map_occs[None]))

    def codeLengthPTop(self, deltaT, EC_za=None, nid=0):
        """
        L(p) = log ( (∆(S) − σ(E))   /  (r−1) ⌋) where σ(E) = sum(E)

        Parameters
        ----------
        deltaT : int
            time delta in the full sequence
        EC_za : int, default=None
            sum of errors
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            float
        L(P)
        """
        if EC_za is None:  # "bare"
            EC_za = 0
        maxv = np.floor((deltaT - EC_za) / (self.nodes[nid]["r"] - 1.))
        clP = np.log2(maxv)
        return clP

    def codeLengthT0(self, deltaT, EC_za=None, nid=0):
        """
        L(τ) = log(∆(S) − σ(E) − (r − 1)p + 1)

        Parameters
        ----------
        deltaT : int
            time delta in the full sequence
        EC_za : int, default=None
            sum of errors
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            float
        L(τ)
        """
        if EC_za is None:  # "bare"
            EC_za = 0

        if OPT_TO:
            maxv = deltaT - EC_za - self.nodes[nid]["p"] * (self.nodes[nid]["r"] - 1) + 1
        else:
            maxv = deltaT + 1
        if EC_za is None and maxv <= 0:  # log2 domain : all positive real numbers
            maxv = 1
        clT = np.log2(maxv)
        return clT

    def hasNestedPDs(self, nid=0):
        """
        Does this pattern has nested periods and/or inter-block distances?

        Parameters
        ----------
        nid : int, default=0
            Node id from which the computation is done

        Returns
        -------
            bool
        If true, this pattern has nested periods and/or inter-block distances else not.
        """
        if len(self.nodes[nid]["children"]) > 1:  # inter-block distances
            return True
        elif len(self.nodes[nid]["children"]) == 1:
            return self.isInterm(self.nodes[nid]["children"][0][0])  # nested
        return False

    def codeLengthPDs(self, Tmax, nid=0, rep=False):
        """
        FIXME : to be explained

        Should the value of Tmax used be the deducted value, or the computed one, which needs to be transmitted first?

        Parameters
        ----------
        Tmax
        nid : int, default=0
            Node id from which the computation is done
        rep

        Returns
        -------

        """
        cl = 0
        if nid not in self.nodes or "children" not in self.nodes[nid]:
            return cl

        # if no interleaving, one repetition cannot span more than 1/r of the parent span
        Tmax_rep = Tmax / self.nodes[nid]["r"]
        if self.allow_interleaving:
            # if interleaving, one repetition can span at most the parent span
            Tmax_rep = Tmax - self.nodes[nid]["r"] + 1.
        # if no interleaving can span at most the period
        elif Tmax_rep > self.nodes[nid]["p"]:
            Tmax_rep = self.nodes[nid]["p"]

        if rep:  # If Tmax_rep provided rather than Tmax, compute Tmax
            Tmax_rep = Tmax
            Tmax = Tmax_rep * self.nodes[nid]["r"]
            if self.allow_interleaving:
                Tmax = Tmax_rep + self.nodes[nid]["r"] - 1.

        if nid > 0 and self.nodes[nid]["r"] > 0:
            if Tmax / (self.nodes[nid]["r"] - 1) < self.nodes[nid]["p"]:
                pdb.set_trace()
                print("PROBLEM!! INCORRECT UPPER BOUND")
            # block period (only not for the root one, already specified)
            pmax = np.floor(Tmax / (self.nodes[nid]["r"] - 1.))
            clp = np.log2(pmax)
            cl += clp

        # inter-blocks distances
        # the distance between two blocks cannot be more than the time spanned by a repetition,
        # there are |children|-1 of them to transmit
        if len(self.nodes[nid]["children"]) > 1:
            cld_i = np.log2(Tmax_rep + 1)
            cld = (len(self.nodes[nid]["children"]) - 1) * cld_i
            ds = [v[1] for v in self.nodes[nid]["children"][1:]]
            cl += cld

        sum_spans = np.sum([nn[1] for nn in self.nodes[nid]["children"][1:]])
        cumsum_spans = 0
        for ni, nn in enumerate(self.nodes[nid]["children"]):
            if self.allow_interleaving:
                if ni > 0:
                    cumsum_spans += nn[1]
                Tmax_i = Tmax_rep - cumsum_spans
            else:
                if ni + 1 == len(self.nodes[nid]["children"]):
                    # last child
                    Tmax_i = Tmax_rep - sum_spans
                else:
                    Tmax_i = self.nodes[nid]["children"][ni + 1][1]
            cl += self.codeLengthPDs(Tmax_i, nn[0])
        return cl

    def codeLength(self, t0, E, data_details, match=None, nid=0):
        """
        FIXME : to be explained

        Parameters
        ----------
        t0
        E
        data_details
        match
        nid

        Returns
        -------

        """
        occsStar = self.getOccsStar(nid=nid, time=t0)
        o_za = self.getLeafKeyFromKey([(-1, self.nodes[0]["r"] - 1)])
        EC_zz = 0
        if E is not None:
            Ed = getEDict(occsStar, E)
            EC_za = self.getCCorr(o_za, Ed)
            clE = codeLengthE(E)
        else:
            Ed = {}
            EC_za = None
            clE = 0

        clEv = self.codeLengthEvents(data_details["adjFreqs"], nid=nid)

        clRs = self.codeLengthR(data_details["nbOccs"], nid=nid)
        clP0 = self.codeLengthPTop(data_details["deltaT"], EC_za, nid=nid)
        clT0 = self.codeLengthT0(data_details["deltaT"], EC_za, nid=nid)

        clPDs = 0.
        if self.hasNestedPDs():
            if E is not None:
                o_zz = self.getLeafKeyFromKey(
                    [(-1, self.nodes[0]["r"] - 1)], cid=-1, rep=-1)
                EC_zz = self.getCCorr(o_zz, Ed)

            Tmax_rep = data_details["t_end"] - t0 - self.nodes[0]["p"] * (self.nodes[0]["r"] - 1.)
            if not self.allow_interleaving:
                Tmax_rep -= EC_zz
                if self.nodes[0]["p"] < Tmax_rep:
                    Tmax_rep = self.nodes[0]["p"]
            else:
                if E is not None:
                    rhks = [self.getKeyFromNid(k, -1)
                            for k in self.getNidsRightmostLeaves()]
                    EC_zz = np.min([self.getCCorr(o, Ed) for o in rhks])
                Tmax_rep -= EC_zz

            if self.transmit_Tmax:
                tmpd = dict([(k[2], k[0]) for k in occsStar])
                Tmax_rep_val = np.max(list(tmpd.values())) - tmpd[o_za]
                clPDs = self.codeLengthPDs(Tmax_rep_val, nid=nid, rep=True)
                if Tmax_rep < 0:
                    if E is None:
                        Tmax_rep = 0
                    else:
                        print("PROBLEM!! INCORRECT TMAX_REP")
                clPDs += np.log2(Tmax_rep + 1)
            else:
                clPDs = self.codeLengthPDs(Tmax_rep, nid=nid, rep=True)

        # L(C) = L(α)+L(r)+L(p)+L(τ0 )+L?+L(E)
        return clEv + clRs + clP0 + clT0 + clPDs + clE

