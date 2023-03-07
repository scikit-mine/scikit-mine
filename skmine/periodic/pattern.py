import copy
import pdb
import re

import numpy as np

from .class_patterns import l_to_key, key_to_l, SUB_SEP, SUP_SEP, OPT_TO


def getEDict(oStar, E=[]):
    """
    Construct a dictionary that maps each unique identifier in the input list of tuples `oStar` to a corresponding value
    in the list `E`.

    Parameters
    ----------
    oStar : list
        A list of tuples representing objects composed of three items.
    E : list, default=[]
        A list of values of shift corrections that correspond to the unique identifiers in `oStar`.
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
    Constructs the list of errors
    # TODO : WARNING! WRONG, this is using absolute errors...
    Parameters
    ----------
    map_occs
    occs

    Returns
    -------

    """
    return [(t - map_occs.get(oid, t)) for (t, alpha, oid) in occs]


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
    LOG_DETAILS = 0  # 1: basic text, 2: LaTeX table

    transmit_Tmax = True
    allow_interleaving = True
    # does overlapping count as interleaved?
    overlap_interleaved = False

    @classmethod
    def parseTreeStr(cls, tree_str, leaves_first=False, from_inner=False):
        MATCH_IN = "\((?P<inner>.*)\)"
        MATCH_PR = "\[r=(?P<r>[0-9]+) p=(?P<p>[0-9]+)\]"
        if leaves_first:
            tmp = re.match(MATCH_IN + MATCH_PR + "$", tree_str)
        else:
            tmp = re.match(MATCH_PR + MATCH_IN + "$", tree_str)

        if tmp is not None:
            patt = cls.parseTreeStr(tmp.group("inner"), leaves_first)
            patt.repeat(int(tmp.group("r")), int(tmp.group("p")))
            return patt
        elif not from_inner:
            return cls.parseInnerStr(tree_str, leaves_first)
        return None

    @classmethod
    def parseInnerStr(cls, inner_str, leaves_first=False):
        MATCH_D = "\[d=(?P<d>[0-9]+)\]"
        pos_L = 0
        d_prev = 0
        parentT = Pattern()
        for tmp in re.finditer(MATCH_D, inner_str):
            T = cls.parseTreeStr(
                inner_str[pos_L: tmp.start()].strip(), leaves_first, from_inner=True)
            if T is None:
                parentT.append(inner_str[pos_L: tmp.start()].strip(), d_prev)
            else:
                parentT.merge(T, d_prev)
            d_prev = int(tmp.group("d"))
            pos_L = tmp.end()

        T = cls.parseTreeStr(
            inner_str[pos_L:].strip(), leaves_first, from_inner=True)
        if T is None:
            parentT.append(inner_str[pos_L:].strip(), d_prev)
        else:
            parentT.merge(T, d_prev)
        return parentT

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
        pc = Pattern()
        pc.next_id = self.next_id
        pc.nodes = copy.deepcopy(self.nodes)
        return pc

    def mapEvents(self, map_evts):
        for nid in self.nodes.keys():
            if self.isLeaf(nid):
                self.nodes[nid]["event"] = map_evts[self.nodes[nid]["event"]]

    def getTranslatedNodes(self, offset):
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
        if not self.isInterm(anchor):
            anchor = 0
        nodes, self.next_id, map_nids = patt.getTranslatedNodes(self.next_id)
        nodes[map_nids[0]]["parent"] = anchor
        self.nodes[anchor]["children"].append((map_nids[0], d))
        self.nodes.update(nodes)
        return map_nids

    def append(self, event, d, anchor=0):
        if not self.isInterm(anchor):
            anchor = 0
        self.nodes[self.next_id] = {"parent": anchor, "event": event}
        self.nodes[anchor]["children"].append((self.next_id, d))
        self.next_id += 1

    def repeat(self, r, p):
        if "r" not in self.nodes[0]:
            self.nodes[0]["p"] = p
            self.nodes[0]["r"] = r
            return
        self.nodes[0]["parent"] = 0
        self.nodes[self.next_id] = self.nodes.pop(0)
        if "children" in self.nodes[self.next_id]:
            for (nc, nd) in self.nodes[self.next_id]["children"]:
                self.nodes[nc]["parent"] = self.next_id
        self.nodes[0] = {"parent": None, "p": p,
                         "r": r, "children": [(self.next_id, 0)]}
        self.next_id += 1

    def isNode(self, nid):
        return nid in self.nodes

    def isInterm(self, nid):
        return self.isNode(nid) and "children" in self.nodes[nid]

    def isLeaf(self, nid):
        return self.isNode(nid) and "children" not in self.nodes[nid]

    def getNidsRightmostLeaves(self, nid=0, rightmost=True):
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
        """
        return np.sum([Ed[k]] + [Ed[kk] for kk in self.gatherCorrKeys(k)])

    def getOccs(self, oStar, t0, E=[]):
        """
        Get the list of timestamp from the pattern reconstructed from his tree after correction.
        """
        if type(E) is dict:
            Ed = E
        else:
            Ed = getEDict(oStar, E)
        return [o[0] + t0 + self.getCCorr(o[-1], Ed) for o in oStar]

    def getCovSeq(self, t0, E=[]):
        """
        Similar as getOccs but returned tuples where the first item correspond to the timestamp after correction
        and the second item is the second parameter of oStar
        """
        oStar = self.getOccsStar()
        # all last elements in the previous tuple associated to his shift correction
        Ed = getEDict(oStar, E)
        return [(o[0] + t0 + self.getCCorr(o[-1], Ed), o[1]) for o in oStar]

    def computeEDict(self, occs):
        refs = {}
        self.getOccsRefs(refs=refs)
        Ed = {}
        t0 = 0
        for nt, (nf, d) in refs.items():
            if nf == "root":
                t0 = occs[nt]
                Ed[nt] = 0
            else:
                Ed[nt] = (occs[nt] - occs[nf]) - d
        return Ed, t0

    def computeEFromO(self, occs):
        occsStar = self.getOccsStar()
        oids = [o[-1] for o in occsStar]
        occsD = dict(zip(*[oids, occs]))
        rEd, rt0 = self.computeEDict(occsD)
        return [rEd[oo] for oo in oids[1:]]

    def getOccsRefs(self, nid=0, pref=[], refs={}, cnref='root', offset=0):
        # for each node indicate which other node is used as time reference, together with perfect offsets
        if not self.isNode(nid):
            return None
        first_occ_cycle = None
        first_occ_rep = None
        next_ref = None
        if self.isInterm(nid):
            for i in range(self.nodes[nid]["r"]):
                for ni, nn in enumerate(self.nodes[nid]["children"]):
                    if ni == 0:  # left-most child
                        if i == 0:  # first rep
                            next_ref = self.getOccsRefs(
                                nn[0], [(ni, i)] + pref, refs, cnref, offset)
                            first_occ_cycle = next_ref
                            first_occ_rep = next_ref
                        else:  # not first rep
                            next_ref = self.getOccsRefs(
                                nn[0], [(ni, i)] + pref, refs, first_occ_rep, self.nodes[nid]["p"])
                            first_occ_rep = next_ref
                    else:  # not left-most child
                        next_ref = self.getOccsRefs(
                            nn[0], [(ni, i)] + pref, refs, next_ref, nn[1])
            return first_occ_cycle
        else:
            current_key = l_to_key(pref[::-1])
            refs[current_key] = (cnref, offset)
            return current_key

    def getNidFromKey(self, k):
        if len(k) == 0:
            return 0
        current_node, level, key_ints = (0, 0, [])
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
        tt = self.getKeyLFromNid(nid, rep)
        if tt is not None and self.isInterm(nid):
            if rep == -1:
                tt.append((-1, self.nodes[nid]["r"] - 1))
            else:
                tt.append((-1, rep))

        if tt is not None:
            return l_to_key(tt)

    def getKeyLFromNid(self, nid=0, rep=0):
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

    def prepareFilter(self, occs, what="fisrtEvt"):
        filt = ".*"
        if what == "fisrtEvt":
            tmp = [b.split(SUB_SEP) for b in occs[0][2].split(SUP_SEP)]
            tmp[0][1] = "[0-9]+"
            filt = "^" + SUP_SEP.join([SUB_SEP.join(tt) for tt in tmp]) + "$"
        elif what == "lastRep":
            tmp = occs[-1][2].split(SUP_SEP)[0].split(SUB_SEP)
            tmp[0] = "[0-9]+"
            filt = "^" + SUB_SEP.join(tmp) + ".*$"
        elif what == "lastRepFirstEvt":
            tlast = occs[-1][2].split(SUP_SEP)[0].split(SUB_SEP)
            tmp = [b.split(SUB_SEP) for b in occs[0][2].split(SUP_SEP)]
            tmp[0][1] = tlast[1]
            filt = "^" + SUP_SEP.join([SUB_SEP.join(tt) for tt in tmp]) + "$"
        return filt

    def filterOccsMatch(self, occs, match_what="fisrtEvt", match=None):
        if match is None:
            match = self.prepareFilter(occs, match_what)
        res = [tt for tt in occs if re.search(match, tt[2])]
        return res

    def getEventsList(self, nid=0, markB=True, map_ev=None):
        if not self.isNode(nid):
            return openB_str + closeB_str
        if self.isInterm(nid):
            ll = []
            for nn in self.nodes[nid]["children"]:
                ll.extend(self.getEventsList(nn[0], markB))
            if markB:
                return ["("] + ll + [")"]
            else:
                return ll
        else:
            return ["%s" % self.nodes[nid]["event"]]

    def getEventsMinor(self, nid=0, rep=False):
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
        Generate the visualization of the cycle
        """
        if not self.isNode(nid):
            return ("\t" * level) + "()\n"

        if self.isInterm(nid):
            ss = "%s|_ [%s] r=%s p=%s\n" % (
                ("\t" * (level)), nid, self.nodes[nid]["r"], self.nodes[nid]["p"])
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
        if not self.isNode(nid):
            return ""
        if self.isInterm(nid):
            return "[%s,%s]" % (self.nodes[nid]["r"], self.nodes[nid]["p"])
        else:
            return "[]"

    def pattMajorKey_list(self, nid=0):
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            return [self.nodes[nid]["r"], self.nodes[nid]["p"]]
        else:
            return []

    def nodeP(self, nid=0):
        if not self.isNode(nid) or not self.isInterm(nid):
            return 0
        else:
            return self.nodes[nid]["p"]

    def nodeR(self, nid=0):
        if not self.isNode(nid) or not self.isInterm(nid):
            return 0
        else:
            return self.nodes[nid]["r"]

    def getMajorOccs(self, occs):
        if self.getDepth() > 1 or self.getWidth() > 1:
            r = self.nodeR(0)
            len_ext_blck = len(occs) // r
            return occs[::len_ext_blck]
        return occs

    # def timeSpanned(self, interleaved=None, nid=0):
    #     # compute the time spanned by a block
    #     # checks whether block is interleaved
    #     if not self.isNode(nid):
    #         return 0
    #     if self.isInterm(nid):
    #         t_ends = []
    #         t_spans = []
    #         cum_ds = 0
    #         for ni, nn in enumerate(self.nodes[nid]["children"]):
    #             if ni > 0:
    #                 cum_ds += nn[1]
    #             t_spans.append(self.timeSpanned(interleaved, nn[0]))
    #             t_ends.append(t_spans[-1] + cum_ds)
    #         tspan = np.max(t_ends)
    #         if interleaved is not None:
    #             if self.overlap_interleaved:  # count overlaps as interleaving
    #                 overlaps = [t_spans[i] >= self.nodes[nid]["children"][i + 1][1]
    #                             for i in range(len(self.nodes[nid]["children"]) - 1)]
    #                 overlaps.append(tspan >= self.nodes[nid]["p"])
    #                 interleaved[nid] = any(overlaps)
    #             else:
    #                 overtaking = [t_spans[i] > self.nodes[nid]["children"][i + 1][1]
    #                               for i in range(len(self.nodes[nid]["children"]) - 1)]
    #                 overtaking.append(tspan > self.nodes[nid]["p"])
    #                 interleaved[nid] = any(overtaking)
    #
    #         return self.nodes[nid]["p"] * (self.nodes[nid]["r"] - 1.) + tspan
    #     else:
    #         return 0
    #
    # def timeSpannedRep(self, nid=0):
    #     # compute the time spanned by a repetition
    #     # checks whether block is interleaved
    #     if not self.isNode(nid):
    #         return 0
    #     if self.isInterm(nid):
    #         t_ends = []
    #         t_spans = []
    #         cum_ds = 0
    #         for ni, nn in enumerate(self.nodes[nid]["children"]):
    #             if ni > 0:
    #                 cum_ds += nn[1]
    #             t_spans.append(self.timeSpanned(nid=nn[0]))
    #             t_ends.append(t_spans[-1] + cum_ds)
    #         tspan = np.max(t_ends)
    #         return tspan
    #     else:
    #         return 0
    #
    # def isInterleaved(self, nid=0):
    #     interleaved = {}
    #     self.timeSpanned(interleaved, nid=nid)
    #     return any(interleaved.values())

    def factorizeTree(self, nid=0):
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
        """
        return self.getDepth(nid) == 1 and self.getWidth(nid) == 1

    def isNested(self, nid=0):
        """
        Checks whether a tree is nested (depth > 1 and width == 1) or not.
        It also works from a certain node id of the tree.
        """
        return self.getDepth(nid) > 1 and self.getWidth(nid) == 1

    def isConcat(self, nid=0):
        """
        Checks whether a tree is concat (depth == 1 and width > 1) or not.
        It also works from a certain node id of the tree.
        """
        return self.getDepth(nid) == 1 and self.getWidth(nid) > 1

    def getTypeStr(self):
        """
        Get the type of pattern (simple, nested, concat or other)
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
        nid : int
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
        if Pattern.LOG_DETAILS == 1:
            print("r\t >> vals%s bounds=%s\tCL=%s" %
                  (rs, min_occs, ["%.3f" % c for c in clrs]))
        if Pattern.LOG_DETAILS == 2:
            print("$\\Clen_0$ & $%s$ & $\\log(%s)=$ & $%s$ \\\\" %
                  (rs, min_occs, ["%.3f" % c for c in clrs]))
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
        if len(self.nodes[nid]["children"]) > 1:
            return True
        elif len(self.nodes[nid]["children"]) == 1:
            return self.isInterm(self.nodes[nid]["children"][0][0])
        return False

    def codeLengthPDs(self, Tmax, nid=0, rep=False):
        # Should the value of Tmax used be the deducted value,
        # or the computed one, which needs to be transmitted first?

        # WARNING! check
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
            # if pmax <= 0: HERE
            #     pdb.set_trace()
            #     print("PMAX", pmax)
            clp = np.log2(pmax)
            if Pattern.LOG_DETAILS == 1:
                print("p%d\t>> val=%d max=%d\tCL=%.3f" %
                      (nid, self.nodes[nid]["p"], pmax, clp))
            if Pattern.LOG_DETAILS == 2:
                print("$\\Cprd_{%d}$ & $%d$ & $\\log(%d)=$ & $%.3f$ \\\\" % (
                    nid, self.nodes[nid]["p"], pmax, clp))
            cl += clp

        # inter-blocks distances
        # the distance between two blocks cannot be more than the time spanned by a repetition,
        # there are |children|-1 of them to transmit
        if len(self.nodes[nid]["children"]) > 1:
            cld_i = np.log2(Tmax_rep + 1)
            cld = (len(self.nodes[nid]["children"]) - 1) * cld_i
            ds = [v[1] for v in self.nodes[nid]["children"][1:]]
            if Pattern.LOG_DETAILS == 1:
                print("d%d\t>> val=%s max=%d\tCL=%d*%.3f=%.3f" % (nid, ds,
                                                                  Tmax_rep, len(
                    self.nodes[nid]["children"]) - 1, cld_i,
                                                                  cld))
            if Pattern.LOG_DETAILS == 2:
                for kk in range(len(self.nodes[nid]["children"]) - 1):
                    print("$d_{%d}$ & $%d$ & $\\log(%d)=$ & $%.3f$ \\\\" % (
                        nid, self.nodes[nid]["children"][kk + 1][1], Tmax_rep, cld_i))
            cl += cld

        sum_spans = np.sum([nn[1]
                            for nn in self.nodes[nid]["children"][1:]])
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
        if Pattern.LOG_DETAILS == 1:
            print("a\t>>\tCL=%.3f" % clEv)
        if Pattern.LOG_DETAILS == 2:
            print("$\\Cev$ & XX & & $%.3f$ \\\\" % clEv)

        clRs = self.codeLengthR(data_details["nbOccs"], nid=nid)
        clP0 = self.codeLengthPTop(data_details["deltaT"], EC_za, nid=nid)
        clT0 = self.codeLengthT0(
            data_details["deltaT"], EC_za, nid=nid)

        clPDs = 0.
        if self.hasNestedPDs():
            if E is not None:
                o_zz = self.getLeafKeyFromKey(
                    [(-1, self.nodes[0]["r"] - 1)], cid=-1, rep=-1)
                EC_zz = self.getCCorr(o_zz, Ed)

            Tmax_rep = data_details["t_end"] - t0 - \
                       self.nodes[0]["p"] * (self.nodes[0]["r"] - 1.)
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
                if Pattern.LOG_DETAILS == 1:
                    print("Tmax\t>> val=%d max=%d\tCL=%.3f" %
                          (Tmax_rep_val, Tmax_rep, np.log2(Tmax_rep + 1)))
                if Pattern.LOG_DETAILS == 2:
                    print("$\\optspanRep^{*}$ & $%d$ & $\\log(%d+1)=$ & $%.3f$ \\\\" %
                          (Tmax_rep_val, Tmax_rep, np.log2(Tmax_rep + 1)))
                # if Tmax_rep <= 0: # HERE
                #     pdb.set_trace()
                #     print("Tmax_rep", Tmax_rep)
                if Tmax_rep < 0:
                    if E is None:
                        Tmax_rep = 0
                    else:
                        print("PROBLEM!! INCORRECT TMAX_REP")
                clPDs += np.log2(Tmax_rep + 1)
            else:
                clPDs = self.codeLengthPDs(Tmax_rep, nid=nid, rep=True)

        # print("CL ev=%.3f rs=%.3f p0=%.3f t0=%.3f pds=%.3f E=%.3f" % (clEv,clRs,clP0,clT0,clPDs,clE))
        # L(C) = L(α)+L(r)+L(p)+L(τ0 )+L?+L(E)
        return clEv + clRs + clP0 + clT0 + clPDs + clE
        # cl = self.codeLengthEvents(data_details["adjFreqs"], nid=nid)
        # cl += self.codeLengthR(data_details["nbOccs"], nid=nid)
        # cl += self.codeLengthPTop(data_details["deltaT"], EC_za, nid=nid)
        # cl += self.codeLengthT0(data_details["deltaT"], EC_za, nid=nid)
        # cl += self.codeLengthPDs(Tmax, nid=nid)
        # cl += self.codeLengthE(E, nid=nid)

        # return cl
