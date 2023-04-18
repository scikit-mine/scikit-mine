import numpy as np

from .class_patterns import computePeriod, computeE, computeLengthCycle, PROPS_MIN_R, PROPS_MAX_OFFSET
from .pattern import Pattern


class Candidate(object):
    """
    Represents a candidate pattern with its properties and methods for working with it.

    Parameters
    ----------
    cid : str
        ID of the candidate.
    P : dict or Pattern
        Dictionary containing the properties of the pattern, or an instance of `Pattern`.
    O : list, optional
        List of corrected occurrences/timestamps of the pattern.
    E : list, optional
        List of cycle shift corrections
    cost : float, optional
        Computed cost of the pattern.

    Attributes
    ----------
    uncov : None or float
        Fraction of the dataset that is not covered by the pattern.
    ev_occ : None or list of tuple
        List of tuples of event occurrences and their time stamps.
    """

    def __init__(self, cid, P, O=None, E=None, cost=None):
        self.cid = cid
        if type(P) is dict and O is None:
            self.initFromDict(P)
        else:
            self.P = P
            self.O = O
            self.E = E
            self.cost = cost
        self.uncov = None
        self.ev_occ = None

    def __str__(self):
        if self.getCost() is None:
            return "\t%s t0=%d" % (self.P, self.getT0())
        if self.getNbUOccs() != self.getNbOccs():
            return "%f/%d=%f >>>(%d)\t%s t0=%d" % (
                self.getCost(), self.getNbUOccs(), self.getCostRatio(), self.getNbOccs(), self.P, self.getT0())
        return "%f/%d=%f (%d)\t%s t0=%d" % (
            self.getCost(), self.getNbUOccs(), self.getCostRatio(), self.getNbOccs(), self.P, self.getT0())

    def initFromDict(self, pdict):
        self.O = pdict.pop("occs")
        self.E = pdict.pop("E", None)
        self.cost = pdict.pop("cost", None)
        if "P" in pdict:
            self.P = pdict["P"]
        else:
            self.P = pdict
            if self.P.get("p") is None:
                self.P["p"] = computePeriod(self.O)
            if self.E is None:
                self.E = computeE(self.O, self.P["p"])

    def getPattT0E(self):
        if self.isPattern():
            P = self.getPattern()
        else:
            P = self.preparePatternSimple()
        return P, self.getT0(), self.E

    def preparePatternSimple(self):
        r0 = len(self.O)
        p0 = self.P["p"]
        if p0 is None:
            p0 = computePeriod(self.O)
        tree = {0: {'p': p0, 'r': r0, 'children': [(1, 0)], 'parent': None},
                1: {'event': self.P["alpha"], 'parent': 0}}
        return Pattern(tree)

    def computeCost(self, data_details, force=False):
        if self.cost is None or force:
            if not self.isPattern():
                self.P = self.preparePatternSimple()
            self.cost = self.P.codeLength(self.getT0(), self.E, data_details)
        return self.cost

    def isPattern(self):
        return type(self.P) is Pattern

    def isComplex(self):
        return self.isPattern() and (self.P.getDepth() > 1 or self.P.getWidth() > 1)

    def setId(self, cid):
        self.cid = cid

    def getId(self):
        return self.cid

    def getPattern(self):
        if self.isPattern():
            return self.P

    def getE(self):
        return self.E

    def getOccs(self):
        return self.O

    def getT0(self):
        return self.O[0]

    def getEvOccs(self):
        if self.isPattern():
            if self.ev_occ is None:
                self.ev_occ = list(
                    zip(*[self.O, self.getEventsMinor(rep=True)]))
            return self.ev_occ
        else:
            return [(o, self.P["alpha"]) for o in self.O]

    def getEventsMinor(self, rep=False):
        if self.isPattern():
            return self.P.getEventsMinor(rep=rep)
        else:
            return self.P["alpha"]

    def getEvent(self):
        """

        Returns
        -------
            list or int or str
        list : if the candidate is a `Pattern` and if there is multiple events, it returns the list of events
        str : if the candidate is a `Pattern` and if there is only one event, it returns the event id as a str
        int : otherwise it returns an int
        """
        if self.isPattern():
            tmp = self.P.getEventsList(add_delimiter=False)
            if len(tmp) == 1:
                return tmp[0]
            else:
                return tmp
        else:
            return self.P["alpha"]

    def getEventTuple(self):
        """
        Get the list of events associated to the candidate

        Returns
        -------
        tuple
            A tuple with all events. If the candidate is a pattern, events are returned as str else as int
        """
        if self.isPattern():
            return tuple(self.P.getEventsList(add_delimiter=False))
        else:
            return tuple([self.P["alpha"]])

    def getNbOccs(self):
        if self.isPattern():
            return self.P.getNbOccs()
        else:
            return len(self.O)

    def getNbUOccs(self):
        if self.isPattern():
            return len(set(self.getEvOccs()))
        else:
            return len(self.O)

    def getMinorKey(self):
        if self.isPattern():
            return self.P.pattMinorKey()
        else:
            return self.P["alpha"]

    def getMajorKey(self):
        if self.isPattern():
            MK = self.P.pattMajorKey()
        else:
            MK = "[%s,%s]" % (len(self.O), self.P["p"])
        return self.getT0(), MK

    def getMajorP(self):
        if self.isPattern():
            return self.P.nodeP(0)
        else:
            return self.P["p"]

    def getMajorR(self):
        if self.isComplex():
            return self.P.nodeR(0)
        else:
            return len(self.O)

    def getMajorO(self):
        if self.isComplex():
            return self.P.getMajorOccs(self.O)
        else:
            return self.O

    def getMajorE(self):
        if self.isComplex():
            return self.P.getMajorOccs([0] + self.E)[1:]
        else:
            return self.E

    def getBlocksO(self, from_rep=0):
        r = self.getMajorR()
        len_ext_blck = len(self.O) // r
        return [self.O[i * len_ext_blck:(i + 1) * len_ext_blck] for i in range(from_rep, r)]

    def getBlocksE(self, from_rep=0):
        r = self.getMajorR()
        len_ext_blck = len(self.O) // r
        tmpE = [0] + self.E
        return [tmpE[i * len_ext_blck + 1:(i + 1) * len_ext_blck] for i in range(from_rep, r)]

    def getTranslatedPNodes(self, offset):
        if self.isPattern():
            nodes, offset, nmap = self.P.getTranslatedNodes(offset)
        else:
            nmap = {0: offset, 1: offset + 1}
            nodes = {offset: {"children": [(offset + 1, 0)]},
                     offset + 1: {"parent": offset, "event": self.P["alpha"]}}
            offset += 2
        return nodes, offset, nmap

    def getCostNoE(self, data_details):
        if self.isPattern():
            return self.P.codeLength(t0=self.getT0(), E=None, data_details=data_details)
        else:
            return computeLengthCycle(data_details, {"alpha": self.getEvent(), "occs": self.O, "p": self.P["p"]},
                                      no_err=True)

    def getCost(self):
        return self.cost

    def getCostRatio(self):
        c = self.getCost()
        if c is None:
            return 0
        return c / self.getNbUOccs()

    def satisfiesMaxCountCover(self, counts_cover, max_o=2):
        for x in self.getEvOccs():
            if counts_cover.get(x, 0) <= max_o:
                return True
        return False

    def updateCountCover(self, counts_cover):
        for x in self.getEvOccs():
            counts_cover[x] = counts_cover.get(x, 0) + 1
        return counts_cover

    def initUncovered(self):
        """
        Initializes the list of occurrences of the candidate pattern that are not covered by occurrences of other
        patterns already selected.

        The set "uncovered" corresponds to "occs(P)\\ O", initially no pattern is selected, O is empty, none of the
        occurrences are covered. See : https://arxiv.org/pdf/1807.01706.pdf#page=29

        Returns
        -------
        None
        """
        self.uncov = set(self.getEvOccs())

    def getUncovered(self):
        return self.uncov

    def getNbUncovered(self):
        if self.uncov is not None:
            return len(self.uncov)
        return -1

    def updateUncovered(self, cover):
        if self.uncov is not None:
            self.uncov.difference_update(cover)
            return len(self.uncov)
        return -1

    def getCostUncoveredRatio(self):
        if self.getNbUncovered() == 0:
            return float("Inf")
        return self.getCost() / self.getNbUncovered()

    def isEfficient(self, dcosts):
        return (self.getCost() / self.getNbUncovered()) < np.mean([dcosts[unc[1]] for unc in self.uncov])

    def getProps(self, nkey=0, max_offset=None):
        if max_offset is None:
            max_offset = PROPS_MAX_OFFSET
        Pp, Pr = (self.getMajorP(), self.getMajorR())
        majO = self.getMajorO()
        majE = self.getMajorE()
        if (PROPS_MIN_R <= 0) or (Pr > PROPS_MIN_R):
            return [(ooe, Pp, Pr, ooi, np.sum(np.abs(majE[ooi:])), nkey, self.getId()) for ooi, ooe in
                    enumerate(majO[:max_offset])]
        else:
            return []

    def factorizePattern(self):
        """
        Factorize the candidate pattern

        The candidate cid is affected to -1.

        Returns
        -------
            list
        List of factorized candidate
        """
        fs = []
        if self.isPattern():
            nf = self.P.canFactorize()
            for t in nf:
                Q = self.P.copy()
                Q.factorizeTree(t)

                refs_P = [c[:2] for c in self.P.getTimesNidsRefs()]
                if len(set(refs_P)) < len(refs_P):
                    # too complex interleaving
                    return fs
                map_Q = dict([(v, k) for (k, v) in enumerate(
                    [c[:2] for c in Q.getTimesNidsRefs()])])

                Qoccs = [None for _ in range(len(self.O))]
                for i, r in enumerate(refs_P):
                    Qoccs[map_Q[r]] = self.O[i]
                QE = Q.computeEFromO(Qoccs)
                tmp = Candidate(-1, Q, Qoccs, QE)
                fs.append(tmp)
                fs.extend(tmp.factorizePattern())
        return fs
