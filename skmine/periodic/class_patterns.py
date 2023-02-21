import copy
import itertools
import pdb
import re
import json
import numpy

# OPT_TO = False
OPT_TO = True
OPT_EVFR = True  # False
ADJFR_MED = False
# PROPS_MAX_OFFSET = 3
# PROPS_MIN_R = 3
PROPS_MAX_OFFSET = -2
PROPS_MIN_R = 0


def _getChained(listsd, keys=None):
    if keys is None:
        keys = list(listsd.keys())
    return itertools.chain(*[listsd.get(k, []) for k in keys])


def computePeriodDiffs(diffs):
    return int(numpy.floor(numpy.median(diffs)))
    # return int(numpy.ceil(numpy.median(diffs)))


def computePeriod(occs, sort=False):
    if sort:
        occs = sorted(occs)
    return computePeriodDiffs(numpy.diff(occs))
    # dfs = sorted([cycle["occs"][i] - cycle["occs"][i-1] for i in range(1,len(cycle["occs"]))])
    # cp = dfs[len(dfs)/2]


def computeE(occs, p0, sort=False):
    if sort:
        occs = sorted(occs)
    return [(occs[i] - occs[i - 1]) - p0 for i in range(1, len(occs))]


def cost_triple(data_details, alpha, dp, deltaE):
    if (data_details["deltaT"] - deltaE + 1) / 2. - dp < 0:
        print("!!!---- Problem delta", data_details["deltaT"], deltaE, dp)
        pdb.set_trace()

    if alpha in data_details["nbOccs"]:
        cl_alpha_and_r = -numpy.log2(data_details["adjFreqs"][alpha]) + data_details.get(
            "blck_delim", 0) + numpy.log2(data_details["nbOccs"][alpha])
    else:
        # cl_alpha_and_r = -numpy.sum([numpy.log2(data_details["adjFreqs"].get(a, 1)) for a in alpha])+
        cl_alpha_and_r = data_details.get("blck_delim", 0)
        cl_alpha_and_r += numpy.log2(numpy.min(
            [data_details["nbOccs"].get(a, data_details["nbOccs"][-1]) for a in alpha]))

    if OPT_TO:
        cl_t0 = numpy.log2(data_details["deltaT"] - deltaE - 2. * dp + 1)
    else:
        cl_t0 = numpy.log2(data_details["deltaT"] + 1)
    return cl_alpha_and_r + cl_t0 + numpy.log2(numpy.floor((data_details["deltaT"] - deltaE) / 2.)) + 4 + numpy.abs(
        deltaE)


def cost_one(data_details, alpha):
    if alpha in data_details["nbOccs"]:
        # -log2(|alpha|)
        cl_alpha = -numpy.log2(data_details["orgFreqs"][alpha])
    else:
        # -numpy.sum([numpy.log2(data_details["adjFreqs"].get(a, 1)) for a in alpha])  # FIXME: to be commented or not?
        cl_alpha = 0

    # cl_alpha + log2(deltaT + 1)
    return cl_alpha + numpy.log2(data_details["deltaT"] + 1)


def computeLengthEOccs(occs, cp):
    # -1
    return numpy.sum([2 + numpy.abs((occs[i] - occs[i - 1]) - cp) for i in range(1, len(occs))])


def computeLengthCycle(data_details, cycle, print_dets=False, no_err=False):
    cp = cycle.get("p")
    if cp is None:
        cp = computePeriod(cycle["occs"])
    cycle["cp"] = cp
    if no_err:
        E = 0
        dE = 0
    else:
        E = computeLengthEOccs(cycle["occs"], cp)
        dE = numpy.sum([(cycle["occs"][i] - cycle["occs"][i - 1]) -
                        cp for i in range(1, len(cycle["occs"]))])

    if cycle["alpha"] in data_details["nbOccs"]:
        alpha = -numpy.log2(data_details["adjFreqs"]
                            [cycle["alpha"]]) + data_details.get("blck_delim", 0)
        r = numpy.log2(data_details["nbOccs"][cycle["alpha"]])
    else:
        # alpha = -numpy.sum([numpy.log2(data_details["adjFreqs"].get(a, 1)) for a in cycle[
        # "alpha"]])+data_details.get("blck_delim", 0)
        alpha = data_details.get("blck_delim", 0)
        r = numpy.log2(numpy.min([data_details["nbOccs"].get(
            a, data_details["nbOccs"][-1]) for a in cycle["alpha"]]))

    p = numpy.log2(numpy.floor(
        (data_details["deltaT"] - dE) / (len(cycle["occs"]) - 1.)))
    if OPT_TO:
        if data_details["deltaT"] - dE - cp * (len(cycle["occs"]) - 1) + 1 <= 0:
            # to compute noErr cost:
            to = 0
        else:
            to = numpy.log2(data_details["deltaT"] -
                            dE - cp * (len(cycle["occs"]) - 1) + 1)
    else:
        to = numpy.log2(data_details["deltaT"] + 1)

    if print_dets:
        print("cp=%d r=%d\talpha=%f\tr=%f\tp=%f\tto=%f\tE=%f" %
              (cp, len(cycle["occs"]), alpha, r, p, to, E))
    return alpha + r + p + to + E


def computeLengthResidual(data_details, residual):
    # print("\n\t\tresiduals %d %s=%f" % (len(residual["occs"]), residual["alpha"], len(residual["occs"])*(
    # numpy.log2(nbOccs[residual["alpha"]]/nbOccs[-1]) + numpy.log2(deltaT))),)
    residual["cp"] = -1
    return len(residual["occs"]) * cost_one(data_details, residual["alpha"])


def computeLengthRC(data_details, rcs):
    cl = 0.
    for rc in rcs:
        if "p" in rc:
            cl += computeLengthCycle(data_details, rc)
        else:
            cl += computeLengthResidual(data_details, rc)
    return cl


def makeOccsAndFreqs(tmpOccs):
    """
    Return the number of occurrences of each event and the original and adjusted
    frequencies of each event in a sequence.

    Parameters
    ----------
    tmpOccs : dict
        A dictionary containing the number of occurrences of each event in the sequence.

    Returns
    -------
    nbOccs : dict
        A dictionary containing the number of occurrences of each event in the sequence,
        including a special key -1 for the total number of events.
    orgFreqs : dict
        A dictionary containing the original frequency of each event in the sequence.
    adjFreqs : dict
        A dictionary containing the adjusted frequency of each event in the sequence.
    blck_delim : float
        A float representing the block delimiter of the sequence.
    """
    if ADJFR_MED:
        return makeOccsAndFreqsMedian(tmpOccs)
    return makeOccsAndFreqsThird(tmpOccs)


def makeOccsAndFreqsMedian(tmpOccs):
    """
    Return the number of occurrences of each event and the original and adjusted frequencies
    of each event in a sequence using the median frequency of all events as a threshold.

    Parameters
    ----------
    tmpOccs : dict
        A dictionary containing the number of occurrences of each event in the sequence.

    Returns
    -------
    nbOccs : dict
        A dictionary containing the number of occurrences of each event in the sequence,
        including a special key -1 for the total number of events.
    orgFreqs : dict
        A dictionary containing the original frequency of each event in the sequence.
    adjFreqs : dict
        A dictionary containing the adjusted frequency of each event in the sequence.
    blck_delim : float
        A float representing the block delimiter of the sequence.
    """
    nbOccs = dict(tmpOccs.items())
    nbOccs[-1] = numpy.sum(list(nbOccs.values())) * 1.

    symOccs = dict(tmpOccs.items())
    med = numpy.median(list(tmpOccs.values()))
    symOccs["("] = med
    symOccs[")"] = med
    symSum = numpy.sum(list(symOccs.values())) * 1.
    adjFreqs = {}
    orgFreqs = {}
    for k in symOccs.keys():
        if OPT_EVFR:
            adjFreqs[k] = symOccs[k] / symSum
            orgFreqs[k] = nbOccs.get(k, 0.) / nbOccs[-1]
        else:
            adjFreqs[k] = 1. / len(symOccs)
            orgFreqs[k] = 1. / len(tmpOccs)
    # print("ADJ FRQ:",  adjFreqs)
    return nbOccs, orgFreqs, adjFreqs, -(numpy.log2(adjFreqs["("]) + numpy.log2(adjFreqs[")"]))


def makeOccsAndFreqsThird(tmpOccs):
    """
    Calculates the occurrence and frequency of each event in the input sequence, using the third method.

    Parameters
    ----------
    tmpOccs : dict
        A dictionary containing the number of occurrences for each event in the sequence.

    Returns
    -------
    nbOccs : dict
        A dictionary containing the number of occurrences of each event in the sequence,
        including a special key -1 for the total number of events.
    orgFreqs : dict
        A dictionary containing the original frequency of each event in the sequence.
    adjFreqs : dict
        A dictionary containing the adjusted frequency of each event in the input sequence.
    blck_delim : float
        The block delimiter calculated as the negated sum of the logarithm base 2 of the adjusted frequency of the
        "(" and ")" events.
    """
    nbOccs = dict(tmpOccs.items())
    nbOccs[-1] = numpy.sum(list(nbOccs.values())) * \
        1.  # -1 is the key for the total number of events in the sequence

    if OPT_EVFR:
        adjFreqs = {"(": 1. / 3, ")": 1. / 3}
    else:
        adjFreqs = {"(": 1. / (len(tmpOccs) + 2), ")": 1. / (len(tmpOccs) + 2)}
    orgFreqs = {}
    for k in nbOccs.keys():
        if k != -1:
            if OPT_EVFR:
                adjFreqs[k] = nbOccs[k] / (3 * nbOccs[-1])
                orgFreqs[k] = nbOccs[k] / nbOccs[-1]
            else:
                adjFreqs[k] = 1. / (len(tmpOccs) + 2)
                orgFreqs[k] = 1. / len(tmpOccs)

    return nbOccs, orgFreqs, adjFreqs, -(numpy.log2(adjFreqs["("]) + numpy.log2(adjFreqs[")"]))


# the keys are a SUP_SEP separated list of SUB_SEP separated (child_id, rep_id) pairs
# both child_id and rep_id start at 0
SUP_SEP = ";"
SUB_SEP = ","


def key_to_l(key):
    try:
        return [list(map(int, b.split(SUB_SEP))) for b in key.split(SUP_SEP)]
    except:
        return []


def l_to_key(l):
    return SUP_SEP.join([("%d" + SUB_SEP + "%d") % tuple(pf) for pf in l])


def l_to_br(l):
    return "B" + ",".join(["%d" % (int(pf[0]) + 1) for pf in l]) + "<" + ",".join(
        ["%d" % (int(pf[1]) + 1) for pf in l]) + ">"


def key_to_br(key):
    l = key_to_l(key)
    return l_to_br(l)


class DataSequence(object):
    """
    A class that constructs a data sequence from a list or a dictionary of events and their respective timestamps.

    Parameters
    ----------
    seq : list or dict
        The input sequence of events and timestamps.
        If `seq` is a list, it should contain tuples of (t, ev) where `t` is the timestamp and `ev` is the event name.
        If `seq` is a dictionary, it should have the event names as keys and the corresponding values should be a list
        of timestamps.

    Attributes
    ----------
    evStarts : dict
        A dictionary with the starting timestamps for each event.
    evEnds : dict
        A dictionary with the ending timestamps for each event.
    data_details : dict
        A dictionary containing information about the data, including the starting timestamp, ending timestamp,
        delta time, number of occurrences, original frequencies, adjusted frequencies and block delimiter.
    seqd : dict
        A dictionary where the keys are event numbers (integers) and the values are lists of timestamps.
    seql: list
        A list of tuples where the first element is the timestamp and the second element is the event number.
    list_ev : list
        A sorted list of unique event names.
    map_ev_num : dict
        A dictionary where the keys are event names and the values are event numbers.

    Examples
    --------
    Creating a `DataSequence` object with a list:

    >>> data = [(1, 'A'), (2, 'B'), (3, 'A'), (4, 'C')]
    >>> ds = DataSequence(data)
    """

    def __init__(self, seq):
        evNbOccs, evStarts, evEnds = ({}, {}, {})
        self.seqd = {}
        self.seql = []
        self.map_ev_num = {}
        self.list_ev = []

        seq_tmp = seq
        if type(seq) is list:
            seq_tmp = {}
            for (t, ev) in seq:
                if ev not in seq_tmp:
                    seq_tmp[ev] = []
                seq_tmp[ev].append(t)

        # construct list and dict, translating ev to num
        self.list_ev = sorted(seq_tmp.keys())
        self.map_ev_num = dict([(v, k) for (k, v) in enumerate(self.list_ev)])
        for q, dt in seq_tmp.items():
            self.seqd[self.map_ev_num[q]] = dt
        for (ev, ts) in self.seqd.items():
            self.seql.extend([(t, ev) for t in ts])
        self.seql.sort()

        for ev, sq in self.seqd.items():
            evNbOccs[ev] = len(sq)
            evStarts[ev] = numpy.min(sq)
            evEnds[ev] = numpy.max(sq)

        nbOccs, orgFreqs, adjFreqs, blck_delim = makeOccsAndFreqs(evNbOccs)
        t_end, t_start = (0, 0)
        if sum(evNbOccs.values()) > 0:
            t_end = numpy.max(list(evEnds.values()))
            t_start = numpy.min(list(evStarts.values()))
        deltaT = t_end - t_start
        self.evStarts = evStarts
        self.evEnds = evEnds
        self.data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
                             "nbOccs": nbOccs, "orgFreqs": orgFreqs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}

    def getInfoStr(self):
        if self.data_details["nbOccs"][-1] == 0:
            ss = "-- Empty Data Sequence"
        else:
            ss = "-- Data Sequence |A|=%d |O|=%d dT=%d (%d to %d)" % (len(
                self.data_details["nbOccs"]) - 1, self.data_details["nbOccs"][-1], self.data_details["deltaT"],
                self.data_details["t_start"],
                self.data_details["t_end"])
            ss += "\n\t" + "\n\t".join(["%s [%d] (|O|=%d f=%.3f dT=%d)" % (
                self.list_ev[k], k, self.data_details["nbOccs"][k], self.data_details["orgFreqs"]
                [k], self.evEnds[k] - self.evStarts[k]) for k in
                sorted(range(len(self.list_ev)), key=lambda x: self.data_details["nbOccs"][x])])
        return ss

    def getEvents(self):
        return self.list_ev

    def getNumToEv(self):
        return dict(enumerate(self.list_ev))

    def getEvToNum(self):
        return self.map_ev_num

    def getSequenceStr(self, sep_te=" ", sep_o="\n", ev=None):
        if ev is None:
            return sep_o.join([("%s" + sep_te + "%s") % (t, self.list_ev[e]) for (t, e) in sorted(self.seql)])
        else:
            if ev in self.map_ev_num:
                return sep_o.join(["%s" % p for p in sorted(self.seqd.get(self.map_ev_num[ev], []))])
            else:
                return sep_o.join(["%s" % p for p in sorted(self.seqd.get(ev, []))])

    def getSequence(self, ev=None):
        if ev is None:
            return self.seql
        else:
            return self.seqd.get(ev, [])

    def getTend(self):
        return self.data_details["t_end"]

    def getDetails(self):
        return self.data_details

    def codeLengthResiduals(self):
        cl = 0
        for ev, ts in self.seqd.items():
            cl += len(ts) * cost_one(self.data_details, ev)
        return cl


class PatternCollection(object):

    def __init__(self, patterns=[]):
        self.patterns = patterns
        self.occ_lists = None

    def __len__(self):
        return len(self.patterns)

    def nbPatternsByType(self):
        nbs = {}
        for (P, t0, E) in self.patterns:
            tstr = P.getTypeStr()
            nbs[tstr] = nbs.get(tstr, 0) + 1
        return nbs

    def setPatterns(self, patterns=[]):
        self.occ_lists = None
        self.patterns = patterns

    def addPatterns(self, patterns):
        return self.patterns.extend(patterns)

    def getPatterns(self):
        return self.patterns

    def getCoveredOccs(self):
        return set().union(*self.getOccLists())

    def getUncoveredOccs(self, data_seq):
        return set(data_seq.getSequence()).difference(*self.getOccLists())

    def getNbUncoveredOccsByEv(self, data_seq):
        nbs = {}
        for (t, ev) in self.getUncoveredOccs(data_seq):
            nbs[ev] = nbs.get(ev, 0) + 1
        return nbs

    def getOccLists(self):
        if self.occ_lists is None:
            self.occ_lists = []
        if len(self.occ_lists) < len(self.patterns):
            for (p, t0, E) in self.patterns[len(self.occ_lists):]:
                self.occ_lists.append(p.getCovSeq(t0, E))
        return self.occ_lists

    def codeLength(self, data_seq):
        cl = 0
        data_details = data_seq.getDetails()
        for (p, t0, E) in self.patterns:
            cl += p.codeLength(t0, E, data_details)
        nbU = self.getNbUncoveredOccsByEv(data_seq)
        for (ev, nb) in nbU.items():
            cl += nb * cost_one(data_details, ev)
        return cl

    def strPatternsTriples(self, data_seq, print_simple=True):
        str_out = ""
        map_ev = data_seq.getNumToEv()
        for pi, (p, t0, E) in enumerate(self.patterns):
            if print_simple or not p.isSimpleCycle():
                Estr = " ".join(["%d" % e for e in E])
                str_out += "%s\t%d\t%s\n" % (p.__str__(map_ev=map_ev,
                                                       leaves_first=True), t0, Estr)
        return str_out

    def strPatternListAndCost(self, data_seq, print_simple=True):
        cl = 0
        data_details = data_seq.getDetails()
        ocls = self.getOccLists()
        map_ev = data_seq.getNumToEv()
        str_out = " ---- COLLECTION PATTERNS\n"
        for pi, (p, t0, E) in enumerate(self.patterns):
            clp = p.codeLength(t0, E, data_details)
            if print_simple or not p.isSimpleCycle():
                str_out += "t0=%d\t%s\tCode length:%f\tsum(|E|)=%d\tOccs (%d/%d)\t%s\n" % (t0, p.__str__(
                    map_ev=map_ev, leaves_first=True), clp, numpy.sum(numpy.abs(E)), len(ocls[pi]), len(set(ocls[pi])),
                    p.getTypeStr())
            # print("P:\tt0=%d\t%s\tCode length:%f\tsum(|E|)=%d\tOccs (%d):%s" % (t0, p, clp, numpy.sum(numpy.abs(
            # E)), len(ocls[pi]), [oo[1] for oo in ocls[pi]]))
            # print("sum(|E|)=%d  E=%s" % (numpy.sum(numpy.abs(E)), E))
            # print("Occs:", ocls[pi], len(ocls[pi]), len(set(ocls[pi])))
            cl += clp
        return str_out, cl

    def strDetailed(self, data_seq,  print_simple=True):
        nbs = self.nbPatternsByType()
        data_details = data_seq.getDetails()

        pl_str, cl = self.strPatternListAndCost(data_seq, print_simple)

        nbs_str = ("Total=%d " % len(self)) + " ".join(
            ["nb_%s=%d" % (k, v) for (k, v) in sorted(nbs.items(), key=lambda x: -x[1])])
        out_str = " ---- COLLECTION STATS (%s)\n" % nbs_str
        nbU = self.getNbUncoveredOccsByEv(data_seq)
        clR = 0
        nbR = 0
        for (ev, nb) in nbU.items():
            nbR += nb
            clR += nb * cost_one(data_details, ev)

        out_str += "Code length patterns (%d): %f\n" % (len(self), cl)
        out_str += "Code length residuals (%d): %f\n" % (nbR, clR)
        cl += clR
        clRonly = data_seq.codeLengthResiduals()
        if clRonly > 0:
            out_str += "-- Total code length = %f (%f%% of %f)\n" % (
                cl, 100 * cl / clRonly, clRonly)
        return out_str, pl_str

    def output_pattern_list_and_cost(self, data_seq, print_simple=True, auto_time_scale_factor=1):
        cl = 0
        data_details = data_seq.getDetails()
        ocls = self.getOccLists()
        map_ev = data_seq.getNumToEv()
        str_out = " ---- COLLECTION PATTERNS\n"
        patterns_list_of_dict = []
        for pi, (p, t0, E) in enumerate(self.patterns):
            dict_pattern = {}

            clp = p.codeLength(t0, E, data_details)
            if print_simple or not p.isSimpleCycle():
                str_out += "t0=%d\t%s\tCode length:%f\tsum(|E|)=%d\tOccs (%d/%d)\t%s\n" % (t0, p.__str__(
                    map_ev=map_ev, leaves_first=True), clp, numpy.sum(numpy.abs(E)), len(ocls[pi]), len(set(ocls[pi])),
                    p.getTypeStr())

            # print("\n pi, t0, E", pi, t0, E)
            # print("p.pattMinorKey()", p.pattMinorKey())
            # print("p.pattMajorKey()_list())", p.pattMajorKey_list())
            # print("p.pattMajorKey()_list()[0])", p.pattMajorKey_list()[0])
            # print("p.pattMajorKey()_list()[1])", p.pattMajorKey_list()[1])
            # print("p.nodeP()", p.nodeP())
            # print("p.nodeR()", p.nodeR())
            # print("p.nodeEv()", p.nodeEv())
            # print("p.pattKey()", p.pattKey())
            # print("p.pattKey()", p.pattKey())
            # print("p.getAllNidKeys", p.getAllNidKeys())
            # print("p.getNidsLeftmostLeaves", p.getNidsLeftmostLeaves())
            # print("p.getNidsRightmostLeaves", p.getNidsRightmostLeaves())
            # print("p.getOccsStar", p.getOccsStar())
            # print("p.getTimesNidsRefs", p.getTimesNidsRefs())
            # print("p.getOccsRefs", p.getOccsRefs())
            # print("p.getKeyFromNid", p.getKeyFromNid())
            # print("p.getKeyLFromNid", p.getKeyLFromNid())
            # print("p.getOccsStarMatch", p.getOccsStarMatch())
            # print("p.getEventsList", p.getEventsList())
            # print("p.getEventsMinor", p.getEventsMinor())
            pattern_tree = p.getTreeDict(map_ev=map_ev)
            pattern_tree["Depth"] = int(p.getDepth())
            pattern_tree["Width"] = int(p.getWidth())
            pattern_tree["auto_time_scale_factor"] = int(
                auto_time_scale_factor)
            pattern_tree = json.dumps(pattern_tree)
            # print("pattern_tree", pattern_tree)
            # print("Ttree", p.getTreeStr())

            # print("p.getCyclePs", p.getCyclePs())
            # print("p.getCycleRs", p.getCycleRs())
            # print("p.getNbLeaves", p.getNbLeaves())
            # print("p.getNbOccs", p.getNbOccs())
            # print("p.getDepth", p.getDepth())
            # print("p.getWidth", p.getWidth())
            # print("p.getAlphabet", p.getAlphabet())
            # print("p.getTypeStr", p.getTypeStr())
            # print("p.getRVals", p.getRVals())
            # print("p str", p.__str__(
            #     map_ev=map_ev, leaves_first=True))

            dict_pattern["t0"] = t0
            dict_pattern["pattern_json_tree"] = pattern_tree
            dict_pattern["pattern_resume"] = p.__str__(
                map_ev=map_ev, leaves_first=True)
            dict_pattern["length_major"] = p.pattMajorKey_list()[0]
            dict_pattern["period_major"] = p.pattMajorKey_list()[1]
            dict_pattern["cost"] = clp
            dict_pattern["type"] = p.getTypeStr()
            # occsStar = p.getOccsStar()
            # Ed = p.getEDict(occsStar, E)
            dict_pattern["E"] = E

            patterns_list_of_dict.append(dict_pattern)

            # print("P:\tt0=%d\t%s\tCode length:%f\tsum(|E|)=%d\tOccs (%d):%s" % (t0, p, clp, numpy.sum(numpy.abs(
            # E)), len(ocls[pi]), [oo[1] for oo in ocls[pi]]))
            # print("sum(|E|)=%d  E=%s" % (numpy.sum(numpy.abs(E)), E))
            # print("Occs:", ocls[pi], len(ocls[pi]), len(set(ocls[pi])))
            cl += clp

        return patterns_list_of_dict, cl

    def output_detailed(self, data_seq, print_simple=True, auto_time_scale_factor=1):
        nbs = self.nbPatternsByType()
        data_details = data_seq.getDetails()

        patterns_list_of_dict, cl = self.output_pattern_list_and_cost(
            data_seq, print_simple, auto_time_scale_factor=auto_time_scale_factor)

        # nbs_str = ("Total=%d " % len(self)) + " ".join(
        #     ["nb_%s=%d" % (k, v) for (k, v) in sorted(nbs.items(), key=lambda x: -x[1])])
        # out_str = " ---- COLLECTION STATS (%s)\n" % nbs_str
        global_stat_dict = {}
        global_stat_dict["Total patterns nb"] = len(self)
        for (k, v) in sorted(nbs.items(), key=lambda x: -x[1]):
            global_stat_dict[k] = v
        nbU = self.getNbUncoveredOccsByEv(data_seq)
        clR = 0
        nbR = 0
        for (ev, nb) in nbU.items():
            nbR += nb
            clR += nb * cost_one(data_details, ev)

        # out_str += "Code length patterns (%d): %f\n" % (len(self), cl)
        # out_str += "Code length residuals (%d): %f\n" % (nbR, clR)
        global_stat_dict[f"Code length patterns"] = cl
        global_stat_dict[f"Code length residuals"] = clR
        global_stat_dict["Total residual nb"] = nbR

        cl += clR
        clRonly = data_seq.codeLengthResiduals()
        if clRonly > 0:
            # out_str += "-- Total code length = %f (%f%% of %f)\n" % (
            #     cl, 100*cl/clRonly, clRonly)
            global_stat_dict["Code length total"] = cl
            global_stat_dict["Total compression ratio"] = 100 * cl / clRonly
            global_stat_dict["Code length residuals only"] = clRonly

        return global_stat_dict, patterns_list_of_dict


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
        children = []
        tmp = None
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

    def getAllNidKeys(self, nid=0, pref=[]):
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            occs = []
            occs.append((nid, l_to_key(pref[::-1])))
            for i in range(self.nodes[nid]["r"]):
                occs.append((nid, l_to_key(pref[::-1] + [(-1, i)])))
                for ni, nn in enumerate(self.nodes[nid]["children"]):
                    occs.extend(self.getAllNidKeys(nn[0], [(ni, i)] + pref))
            return occs
        else:
            return [(nid, l_to_key(pref[::-1]))]

    def getNidsLeftmostLeaves(self, nid=0, leftmost=True):
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            leftmost_nids = []
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                leftmost_nids.extend(
                    self.getNidsLeftmostLeaves(nn[0], ni == 0))
            return leftmost_nids
        else:
            if leftmost:
                return [nid]
            return []

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
        if not self.isNode(nid):
            return []
        if self.isInterm(nid):
            occs = []
            for i in range(self.nodes[nid]["r"]):
                tt = time + i * self.nodes[nid]["p"]
                for ni, nn in enumerate(self.nodes[nid]["children"]):
                    tt += nn[1]
                    occs.extend(self.getTimesNidsRefs(
                        nn[0], [(ni, i)] + pref, tt))
            return occs
        else:
            return [(time, nid, l_to_key(pref[::-1]))]

    def getEDict(self, oStar, E=[]):
        if len(E) >= len(oStar) - 1:
            Ex = [0] + list(E[:len(oStar) - 1])
        else:
            Ex = [0 for o in oStar]
        oids = [o[-1] for o in oStar]
        return dict(zip(*[oids, Ex]))

    def getECDict(self, Ed):
        return dict([(oi, self.getCCorr(oi, Ed)) for oi in Ed.keys()])

    def getCCorr(self, k, Ed):
        return numpy.sum([Ed[k]] + [Ed[kk] for kk in self.gatherCorrKeys(k)])

    def getOccs(self, oStar, t0, E=[]):
        if type(E) is dict:
            Ed = E
        else:
            Ed = self.getEDict(oStar, E)
        return [o[0] + t0 + self.getCCorr(o[-1], Ed) for o in oStar]

    def getCovSeq(self, t0, E=[]):
        oStar = self.getOccsStar()
        Ed = self.getEDict(oStar, E)
        return [(o[0] + t0 + self.getCCorr(o[-1], Ed), o[1]) for o in oStar]

    def getOccsByRefs(self, oStar, t0, E=[]):
        if type(E) is dict:
            Ed = E
        else:
            Ed = self.getEDict(oStar, E)

        refs = {}
        self.getOccsRefs(refs=refs)
        refs_inv = {}
        for k, v in refs.items():
            if v[0] not in refs_inv:
                refs_inv[v[0]] = []
            refs_inv[v[0]].append(k)

        times = {"root": t0}
        border = ["root"]
        while len(border) > 0:
            n = border.pop(0)
            for nt in refs_inv[n]:
                times[nt] = times[n] + refs[nt][-1] + Ed[nt]
                if nt in refs_inv:
                    border.append(nt)
        return [times[o[-1]] for o in oStar]

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
        while level < len(key_ints) and level > -1:
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
            while cid < len(self.nodes[parent]["children"]) and cid > -1:
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
        # (child_id, rep_id)
        # print("--- Gather", k)
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
                # ll = self.getLeafKeyFromKey(key_ints+[(i, last[1])])
                # print("Left sibling %s %s: %s" % (i, key_ints+[(i, last[1])], ll) )
                cks.append(self.getLeafKeyFromKey(key_ints + [(i, last[1])]))
            for i in range(last[1]):
                # ll = self.getLeafKeyFromKey(key_ints+[(-1, i)])
                # print("Previous reps %s %s: %s" % (i, key_ints+[(-1, i)], ll))
                cks.append(self.getLeafKeyFromKey(key_ints + [(-1, i)]))
            cks.extend(self.gatherCorrKeys(key_ints))
            # if len(cks) > 0:
            #     pdb.set_trace()
            #     print(k, cks)
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
        # print("filter occs %s: %s\n\t>>%s" % (match_what, match, res))
        return res

    def getOccsStarMatch(self, match=None, nid=0, pref=[], time=0):
        occs = self.getOccsStar(nid, pref, time)
        return self.filterOccsMatch(occs, match)

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

    def getTreeDict(self, nid=0, map_ev=None):
        """
        A function that constructs recursively the json Tree from the pattern. 

        Parameters
        ----------
        self : class Pattern(object)

        nid : integer
            number of the node of the pattern tree

        map_ev : dict
            A dictionary where the keys are event names and the values are event numbers.

        Returns
        -------
        json string
            representing the pattern tree. 
        """

        if not self.isNode(nid):
            return {}
        if self.isInterm(nid):
            parent = self.nodes[nid]["parent"]
            children = self.nodes[nid]["children"]
            ss = {
                "nid": int(nid),
                "r": int(self.nodes[nid]["r"]),
                "p": int(self.nodes[nid]["p"]),
                "parent": str(parent) if parent == None else parent,
                "children_tuple": ["(nid=" + str(int(nn[0])) + ", d=" + str(int(nn[1])) + ")" for nn in children]
            }

            ss["children"] = []
            for nn in children:
                # ss += "%s| d=%s\n" % (("\t" * (level + 1)), nn[1])
                parent = self.nodes[nn[0]]["parent"]
                child = {
                    "nid": int(nn[0]),
                    "d":   int(nn[1]),
                    "parent": str(parent) if parent == None else parent,
                }
                if self.isInterm(nn[0]):
                    for key, value in self.getTreeDict(nn[0], map_ev=map_ev).items():
                        child[key] = value
                else:
                    child["event"] = self.getTreeDict(nn[0], map_ev=map_ev)
                ss["children"].append(child)
            return ss
        else:
            if map_ev is not None:
                return map_ev.get(self.nodes[nid]["event"], self.nodes[nid]["event"])
                # return "%s|_ [%s] %s\n" % (
                #     ("\t" * level), nid, map_ev.get(self.nodes[nid]["event"], self.nodes[nid]["event"]))
            else:
                return str(self.nodes[nid]["event"])
                # return "%s|_ [%s] %s\n" % (("\t" * level), nid, self.nodes[nid]["event"])

    # trees["P5"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0), (2, 3), (3, 1)], 'parent': None},
    #                1: {'event': 'b', 'parent': 0},
    #                2: {'event': 'a', 'parent': 0},
    #                3: {'event': 'c', 'parent': 0}}

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

    def nodeEv(self, nid=0):
        if not self.isNode(nid) or self.isInterm(nid):
            return None
        else:
            return self.nodes[nid]["event"]

    def getMajorOccs(self, occs):
        if self.getDepth() > 1 or self.getWidth() > 1:
            r = self.nodeR(0)
            len_ext_blck = len(occs) // r
            return occs[::len_ext_blck]
        return occs

    def timeSpanned(self, interleaved=None, nid=0):
        # compute the time spanned by a block
        # checks whether block is interleaved
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            t_ends = []
            t_spans = []
            cum_ds = 0
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                if ni > 0:
                    cum_ds += nn[1]
                t_spans.append(self.timeSpanned(interleaved, nn[0]))
                t_ends.append(t_spans[-1] + cum_ds)
            tspan = numpy.max(t_ends)
            if interleaved is not None:
                if self.overlap_interleaved:  # count overlaps as interleaving
                    overlaps = [t_spans[i] >= self.nodes[nid]["children"][i + 1][1]
                                for i in range(len(self.nodes[nid]["children"]) - 1)]
                    overlaps.append(tspan >= self.nodes[nid]["p"])
                    interleaved[nid] = any(overlaps)
                else:
                    overtaking = [t_spans[i] > self.nodes[nid]["children"][i + 1][1]
                                  for i in range(len(self.nodes[nid]["children"]) - 1)]
                    overtaking.append(tspan > self.nodes[nid]["p"])
                    interleaved[nid] = any(overtaking)

            return self.nodes[nid]["p"] * (self.nodes[nid]["r"] - 1.) + tspan
        else:
            return 0

    def timeSpannedRep(self, nid=0):
        # compute the time spanned by a repetition
        # checks whether block is interleaved
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            t_ends = []
            t_spans = []
            cum_ds = 0
            for ni, nn in enumerate(self.nodes[nid]["children"]):
                if ni > 0:
                    cum_ds += nn[1]
                t_spans.append(self.timeSpanned(nid=nn[0]))
                t_ends.append(t_spans[-1] + cum_ds)
            tspan = numpy.max(t_ends)
            return tspan
        else:
            return 0

    def isInterleaved(self, nid=0):
        interleaved = {}
        self.timeSpanned(interleaved, nid=nid)
        return any(interleaved.values())

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
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return numpy.sum([self.getNbLeaves(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getNbOccs(self, nid=0):
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return self.nodes[nid]["r"] * numpy.sum([self.getNbOccs(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getDepth(self, nid=0):
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return 1 + numpy.max([self.getDepth(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 0

    def getWidth(self, nid=0):
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            return numpy.sum([self.getWidth(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    def getAlphabet(self, nid=0):
        # recursively collects all the different events
        if not self.isNode(nid):
            return set()
        if self.isInterm(nid):
            return set().union(*[self.getAlphabet(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return set([self.nodes[nid]["event"]])

    def isSimpleCycle(self, nid=0):
        return self.getDepth(nid) == 1 and self.getWidth(nid) == 1

    def isNested(self, nid=0):
        return self.getDepth(nid) > 1 and self.getWidth(nid) == 1

    def isConcat(self, nid=0):
        return self.getDepth(nid) == 1 and self.getWidth(nid) > 1

    def getTypeStr(self):
        if self.isSimpleCycle():
            return "simple"
        elif self.isNested():
            return "nested"
        elif self.isConcat():
            return "concat"
        else:
            return "other"

    # L(α) = − log(fr (α)) = − log(∣ ∣S(α)∣ ∣ / |S|)
    def codeLengthEvents(self, adjFreqs, nid=0):
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            return numpy.sum([-numpy.log2(adjFreqs["("]), -numpy.log2(adjFreqs[")"])] +
                             [self.codeLengthEvents(adjFreqs, nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return -numpy.log2(adjFreqs[self.nodes[nid]["event"]])

    def getMinOccs(self, nbOccs, min_occs, nid=0):
        # recursively collects info on the least occurring event in each block
        # to be used to determine the code length for r_X
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            min_r = numpy.min([self.getMinOccs(nbOccs, min_occs, nn[0])
                               for nn in self.nodes[nid]["children"]])
            min_occs.append(min_r)
            return min_r
        else:
            return nbOccs[self.nodes[nid]["event"]]

    def getRVals(self, nid=0):
        # recursively collects info on the least occurring event in each block
        # to be used to determine the code length for r_X
        if not self.isNode(nid):
            return -1
        if self.isInterm(nid):
            rs = [self.nodes[nid]["r"]]
            for nn in self.nodes[nid]["children"]:
                rs.extend(self.getRVals(nn[0]))
            return rs
        else:
            return []

    # L(r) = log(∣ ∣S(α)∣ ∣),
    def codeLengthR(self, nbOccs, nid=0):
        # determine the code length for r_X
        # based on info on the least occurring event in each block
        min_occs = []
        self.getMinOccs(nbOccs, min_occs, nid)
        rs = self.getRVals()
        clrs = numpy.log2(min_occs)
        if Pattern.LOG_DETAILS == 1:
            print("r\t >> vals%s bounds=%s\tCL=%s" %
                  (rs, min_occs, ["%.3f" % c for c in clrs]))
        if Pattern.LOG_DETAILS == 2:
            print("$\\Clen_0$ & $%s$ & $\\log(%s)=$ & $%s$ \\\\" %
                  (rs, min_occs, ["%.3f" % c for c in clrs]))
        return numpy.sum(clrs)

    def cardO(self, nid=0):
        # computes the number of occurrences generated by a pattern
        if not self.isNode(nid):
            return 0
        if self.isInterm(nid):
            return self.nodes[nid]["r"] * numpy.sum([self.cardO(nn[0]) for nn in self.nodes[nid]["children"]])
        else:
            return 1

    # WARNING! WRONG, this is using absolute errors...
    def getEforOccs(self, map_occs, occs):
        # constructs the list of errors
        return [(t - map_occs.get(oid, t)) for (t, alpha, oid) in occs]

    def getE(self, map_occs, nid=0):
        return self.getEforOccs(map_occs, self.getOccsStar(nid, time=map_occs[None]))

    ####

    # L(E) = 2 + ∑_{e∈E} |e| ? ( It should not be     :  2 |E| + ∑_{e∈E} |e|     ? )
    def codeLengthE(self, E, nid=0):
        clE = numpy.sum([2 + numpy.abs(e) for e in E])  # -1
        if Pattern.LOG_DETAILS == 1:
            print("E\t>> nb=%d cumsum=%d\tCL=%.3f" %
                  (len(E), numpy.sum([numpy.abs(e) for e in E]), clE))
        if Pattern.LOG_DETAILS == 2:
            print("$\\Csc$ & $\\LL{%s}$ & & $%.3f$ \\\\" % (E, clE))
        return clE

    # L(p) = log ( (∆(S) − σ(E))   /  (r−1) ⌋) .

    def codeLengthPTop(self, deltaT, EC_za=None, nid=0):
        if EC_za is None:  # "bare"
            EC_za = 0
        maxv = numpy.floor((deltaT - EC_za) / (self.nodes[nid]["r"] - 1.))
        clP = numpy.log2(maxv)
        if Pattern.LOG_DETAILS == 1:
            print("p0\t>> val=%d max=%d\tCL=%.3f" %
                  (self.nodes[nid]["p"], maxv, clP))
        if Pattern.LOG_DETAILS == 2:
            print("$\\Cprd_0$ & $%d$ & $\\log(%d)=$ & $%.3f$ \\\\" %
                  (self.nodes[nid]["p"], maxv, clP))
        return clP

    # L(τ ) = log(∆(S) − σ(E) − (r − 1)p + 1) .
    def codeLengthT0(self, deltaT, EC_za=None, nid=0, t0=0, tstart=0):
        if EC_za is None:  # "bare"
            EC_za = 0

        if OPT_TO:
            maxv = deltaT - EC_za - \
                self.nodes[nid]["p"] * (self.nodes[nid]["r"] - 1) + 1
        else:
            maxv = deltaT + 1
        if EC_za is None and maxv <= 0:
            maxv = 1
        clT = numpy.log2(maxv)
        if EC_za is not None and (t0 - tstart) > maxv:
            pdb.set_trace()
        if Pattern.LOG_DETAILS == 1:
            print("t0\t>> val=%d max=%d\tCL=%.3f" % (t0, maxv, clT))
        if Pattern.LOG_DETAILS == 2:
            print("$\\Cto$ & $%d$ & $\\log(%d)=$ & $%.3f$ \\\\" %
                  (t0, maxv, clT))
        return clT

    def hasNestedPDs(self, nid=0):
        # does this pattern has nested periods and/or inter-block distances?
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
            pmax = numpy.floor(Tmax / (self.nodes[nid]["r"] - 1.))
            # if pmax <= 0: HERE
            #     pdb.set_trace()
            #     print("PMAX", pmax)
            clp = numpy.log2(pmax)
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
            cld_i = numpy.log2(Tmax_rep + 1)
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

        sum_spans = numpy.sum([nn[1]
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

    def codeLengthBare(self, data_details, match=None, nid=0):
        t0 = 0
        clEv = self.codeLengthEvents(data_details["adjFreqs"], nid=nid)
        clRs = self.codeLengthR(data_details["nbOccs"], nid=nid)
        clP0 = self.codeLengthPTop(data_details["deltaT"], None, nid=nid)
        clT0 = self.codeLengthT0(
            data_details["deltaT"], None, nid=nid, t0=t0, tstart=data_details["t_start"])

        clPDs = 0.
        if self.hasNestedPDs():

            Tmax_rep = data_details["t_end"] - t0 - \
                self.nodes[0]["p"] * (self.nodes[0]["r"] - 1.)
            if not self.allow_interleaving:
                if self.nodes[0]["p"] < Tmax:
                    Tmax_rep = self.nodes[0]["p"]

            if self.transmit_Tmax:
                clPDs = self.codeLengthPDs(Tmax_rep, nid=nid, rep=True)
                clPDs += numpy.log2(Tmax_rep)
            else:
                clPDs = self.codeLengthPDs(Tmax_rep, nid=nid, rep=True)

        # print("CL ev=%.3f rs=%.3f p0=%.3f t0=%.3f pds=%.3f E=%.3f" % (clEv,clRs,clP0,clT0,clPDs,clE))
        return clEv + clRs + clP0 + clT0 + clPDs

    def codeLength(self, t0, E, data_details, match=None, nid=0):
        occsStar = self.getOccsStar(nid=nid, time=t0)
        o_za = self.getLeafKeyFromKey([(-1, self.nodes[0]["r"] - 1)])
        EC_zz = 0
        if E is not None:
            Ed = self.getEDict(occsStar, E)
            EC_za = self.getCCorr(o_za, Ed)
            clE = self.codeLengthE(E, nid=nid)
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
            data_details["deltaT"], EC_za, nid=nid, t0=t0, tstart=data_details["t_start"])

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
                    EC_zz = numpy.min([self.getCCorr(o, Ed) for o in rhks])
                Tmax_rep -= EC_zz

            if self.transmit_Tmax:
                tmpd = dict([(k[2], k[0]) for k in occsStar])
                Tmax_rep_val = numpy.max(list(tmpd.values())) - tmpd[o_za]
                clPDs = self.codeLengthPDs(Tmax_rep_val, nid=nid, rep=True)
                if Pattern.LOG_DETAILS == 1:
                    print("Tmax\t>> val=%d max=%d\tCL=%.3f" %
                          (Tmax_rep_val, Tmax_rep, numpy.log2(Tmax_rep + 1)))
                if Pattern.LOG_DETAILS == 2:
                    print("$\\optspanRep^{*}$ & $%d$ & $\\log(%d+1)=$ & $%.3f$ \\\\" %
                          (Tmax_rep_val, Tmax_rep, numpy.log2(Tmax_rep + 1)))
                # if Tmax_rep <= 0: # HERE
                #     pdb.set_trace()
                #     print("Tmax_rep", Tmax_rep)
                if Tmax_rep < 0:
                    if E is None:
                        Tmax_rep = 0
                    else:
                        print("PROBLEM!! INCORRECT TMAX_REP")
                clPDs += numpy.log2(Tmax_rep + 1)
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

        return cl


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
        List of occurrences/timestamps of the pattern.
    E : list, optional
        List of deltaE of the occurrences.
    cost : float, optional
        Computed cost of the pattern.

    Attributes
    ----------
    prop_list : list
        List of the names of the properties of the pattern.
    prop_map : dict
        Dictionary mapping property names to their index in `prop_list`.
    uncov : None or float
        Fraction of the dataset that is not covered by the pattern.
    ev_occ : None or list of tuple
        List of tuples of event occurrences and their time stamps.
    """
    prop_list = ["t0i", "p0", "r0", "offset", "cumEi", "new", "cid"]
    prop_map = dict([(v, k) for k, v in enumerate(prop_list)])

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
        if self.isPattern():
            tmp = self.P.getEventsList(markB=False)
            if len(tmp) == 1:
                return tmp[0]
            else:
                return tmp
        else:
            return self.P["alpha"]

    def getEventTuple(self):
        if self.isPattern():
            return tuple(self.P.getEventsList(markB=False))
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
        return (self.getT0(), MK)

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
        return (self.getCost() / self.getNbUncovered()) < numpy.mean([dcosts[unc[1]] for unc in self.uncov])

    def adjustOccs(self):
        if not self.isPattern() and (self.uncov is not None) and (self.getNbUncovered() < self.getNbOccs()):
            okk = self.getEvOccs()
            mni, mxi = (0, len(self.O) - 1)
            while okk[mni] not in self.uncov:
                mni += 1
            while okk[mxi] not in self.uncov:
                mxi -= 1
            self.P["occs_up"] = [self.O[kk] for kk in range(mni, mxi + 1)]

    def getPropsFirst(self, nkey=0):
        # ["t0i", "p0", "r0", "offset", "cumEi", "new", "cid"]
        return (
            self.getT0(), self.getMajorP(), self.getMajorR(
            ), 0, numpy.sum(numpy.abs(self.getMajorE())), nkey,
            self.getId())

    def getPropsAll(self, nkey=0):
        Pp, Pr = (self.getMajorP(), self.getMajorR())
        majO = self.getMajorO()
        majE = self.getMajorE()
        return [(ooe, Pp, Pr, ooi, numpy.sum(numpy.abs(majE[ooi:])), nkey, self.getId()) for ooi, ooe in
                enumerate(majO[:-2])]

    def getProps(self, nkey=0, max_offset=None):
        if max_offset is None:
            max_offset = PROPS_MAX_OFFSET
        Pp, Pr = (self.getMajorP(), self.getMajorR())
        majO = self.getMajorO()
        majE = self.getMajorE()
        if (PROPS_MIN_R <= 0) or (Pr > PROPS_MIN_R):
            return [(ooe, Pp, Pr, ooi, numpy.sum(numpy.abs(majE[ooi:])), nkey, self.getId()) for ooi, ooe in
                    enumerate(majO[:max_offset])]
        else:
            return []

    def factorizePattern(self):
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

                Qoccs = [None for i in range(len(self.O))]
                for i, r in enumerate(refs_P):
                    Qoccs[map_Q[r]] = self.O[i]
                QE = Q.computeEFromO(Qoccs)
                tmp = Candidate(-1, Q, Qoccs, QE)
                fs.append(tmp)
                fs.extend(tmp.factorizePattern())
        return fs


def propCmp(props, pid):
    if type(props) is list:
        return (props[pid][Candidate.prop_map["t0i"]],
                props[pid][Candidate.prop_map["p0"]],
                props[pid][Candidate.prop_map["r0"]])
    else:
        return (props[pid, Candidate.prop_map["t0i"]],
                props[pid, Candidate.prop_map["p0"]],
                props[pid, Candidate.prop_map["r0"]])


def sortPids(props, pids=None):
    if pids is None:
        pids = range(len(props))
    return sorted(pids, key=lambda x: propCmp(props, x))


def mergeSortedPids(props, pidsA, pidsB):
    i = 0
    while i < len(pidsA) and len(pidsB) > 0:
        if propCmp(props, pidsA[i]) > propCmp(props, pidsB[0]):
            pidsA.insert(i, pidsB.pop(0))
        i += 1
    if len(pidsB) > 0:
        pidsA.extend(pidsB)
    return pidsA


class CandidatePool(object):
    """
    CandidatePool is a class for storing and managing Candidate objects.

    Attributes
    ----------
    candidates : dict
        A dictionary of Candidate objects stored using candidate IDs as keys.
    next_cid : int
        The ID to be used for the next created Candidate.
    cand_props : numpy.ndarray
        A numpy array containing the properties of all candidates. Each row represents a candidate and each column a
        different property. The first property is the "new" property, followed by the properties of the Candidate
        object.
    sorted_pids : list or range object
        A list or range object representing the IDs of all candidates, sorted in order of the "new" property and then
        in descending order of the "score" property.
    map_minorKs : dict
        A dictionary containing a mapping from minor keys to a dictionary that maps major keys to candidate IDs.
    map_nkeys : dict
        A dictionary that maps new keys to indices in the list_nkeys attribute.
    list_nkeys : list
        A list containing all new keys used by candidates.
    new_cids : dict
        A dictionary that maps new keys to a list of candidate IDs that are new according to that key.
    new_minorKs : dict
        A dictionary that maps new keys to a set of minor keys for candidates that are new according to that key.
    """

    def __init__(self, patts=[]):
        self.candidates = {}
        self.next_cid = 0
        self.cand_props = None
        self.sorted_pids = None
        self.map_minorKs = {}
        self.map_nkeys = {}
        self.list_nkeys = []
        self.resetNew()
        if len(patts) > 0:
            self.addCands(patts)

    def resetNew(self):
        """
        Resets the new_cids and new_minorKs attributes to empty dictionaries.
        """
        self.new_cids = {}
        self.new_minorKs = {}

    def getSortedPids(self):
        if self.sorted_pids is None:
            self.sorted_pids = sortPids(self.cand_props)
        return self.sorted_pids

    def isNewPid(self, pid, nkey=None):
        if (nkey in self.map_nkeys) and (self.cand_props is not None) and (pid < self.cand_props.shape[0]):
            return self.cand_props[pid, Candidate.prop_map["new"]] == self.map_nkeys[nkey]
        return False

    def areNewPids(self, pids, nkey=None):
        if (nkey in self.map_nkeys) and (self.cand_props is not None):
            return self.cand_props[pids, Candidate.prop_map["new"]] == self.map_nkeys[nkey]
        return numpy.zeros(len(pids), dtype=bool)

    def getPidsForCid(self, cid):
        return numpy.where(self.cand_props[:, -1] == cid)[0]

    def getCidsForMinorK(self, mK):
        return list(self.map_minorKs.get(mK, {}).values())

    def getCandidate(self, cid):
        return self.candidates[cid]

    def getCandidates(self):
        return self.candidates

    def getNewKNum(self, nkey):
        return self.map_nkeys.get(nkey, -1)

    def getNewPids(self, nkey):
        if (nkey in self.map_nkeys) and (self.cand_props is not None):
            return numpy.where(self.cand_props[:, Candidate.prop_map["new"]] == self.map_nkeys[nkey])[0]
        return []

    def getNewCids(self, nkey):
        return self.new_cids.get(nkey, [])

    def getNewMinorKeys(self, nkey):
        return self.new_minorKs.get(nkey, set())

    def nbNewCandidates(self, nkey=None):
        if nkey is None:
            return numpy.sum([len(v) for v in self.new_cids.values()])
        return len(self.new_cids.get(nkey, []))

    def nbMinorKs(self):
        return len(self.map_minorKs)

    def nbCandidates(self):
        return len(self.candidates)

    def nbProps(self):
        return self.cand_props.shape[0]

    def getPropMat(self):
        return self.cand_props

    def getProp(self, pid, att=None):
        """
        Returns the property values for the specified candidate ID and property name or index.
        """
        if type(att) is int and att < len(Candidate.prop_list):
            return self.cand_props[pid, att]
        elif att in Candidate.prop_map:
            return self.cand_props[pid, Candidate.prop_map[att]]
        return self.cand_props[pid, :]

    def addCand(self, p, nkey=None, with_props=True):
        """
        Adds a new candidate to the candidate pool.

        Parameters
        ----------
        p : Candidate or tuple
            The candidate to add or a tuple to create a new candidate.
        nkey : any type
            The new key of the candidate.
        with_props : bool
            Whether to generate properties for the candidate.

        Returns
        -------
        Candidate or None
            The newly added candidate or None if candidate is a duplicate.
        """
        if type(p) is Candidate:
            c = p
        else:  # create candidate
            c = Candidate(-1, p)

        mK = c.getMinorKey()
        MK = c.getMajorKey()
        if mK in self.map_minorKs:
            if MK in self.map_minorKs[mK]:
                return None

        self.candidates[self.next_cid] = c
        c.setId(self.next_cid)

        # record event
        if mK not in self.map_minorKs:
            self.map_minorKs[mK] = {MK: self.next_cid}
        else:
            self.map_minorKs[mK][MK] = self.next_cid

        # record as new
        if nkey not in self.map_nkeys:
            self.map_nkeys[nkey] = len(self.list_nkeys)
            self.list_nkeys.append(nkey)
        if nkey not in self.new_minorKs:
            self.new_minorKs[nkey] = set()
        self.new_minorKs[nkey].add(mK)
        if nkey not in self.new_cids:
            self.new_cids[nkey] = []
        self.new_cids[nkey].append(self.next_cid)

        # generate properties
        if with_props:
            props = self.candidates[self.next_cid].getProps(
                self.map_nkeys[nkey])
            if self.cand_props is None:
                npids = range(len(props))
                self.cand_props = numpy.array(props)
                self.sorted_pids = range(len(props))
            else:
                npids = range(
                    self.cand_props.shape[0], self.cand_props.shape[0] + len(props))
                self.cand_props = numpy.vstack([self.cand_props, props])
                self.sorted_pids = mergeSortedPids(
                    self.cand_props, self.sorted_pids, npids)

        self.next_cid += 1
        return c

    def addCands(self, ps, nkey=None, costOne=None):
        """
        Adds multiple candidates to the candidate pool.

        Parameters:
        ps (list): A list of candidate patterns.
        nkey (optional): The new key for the candidates, if any.
        costOne (optional): The cost of a substitution operation.
        """
        if nkey not in self.map_nkeys:
            self.map_nkeys[nkey] = len(self.list_nkeys)
            self.list_nkeys.append(nkey)
        props = []
        for p in ps:
            c = self.addCand(p, nkey, with_props=False)
            # HERE figure out reasonable max_offset
            max_offset = None
            if costOne is not None and c.getCost() is not None:
                max_offset = numpy.minimum(
                    int(numpy.floor(c.getCost() / costOne)), c.getNbOccs() - 2)
                # if c.getNbOccs() > 3:
                #     print("NbOccs=%s, maxOff=%s" % (c.getNbOccs(), max_offset))
            if c is not None:
                props.extend(c.getProps(
                    self.map_nkeys[nkey], max_offset=max_offset))

        if len(props) == 0:
            return

        if self.cand_props is None:
            npids = range(len(props))
            self.cand_props = numpy.array(props)
            self.sorted_pids = sortPids(self.cand_props, npids)
        else:
            npids = range(
                self.cand_props.shape[0], self.cand_props.shape[0] + len(props))
            self.cand_props = numpy.vstack([self.cand_props, props])
            self.sorted_pids = mergeSortedPids(
                self.cand_props, self.sorted_pids, sortPids(self.cand_props, npids))


if __name__ == "__main__":
    # p = Pattern("a", 2, 3)
    # p1 = Pattern("b", 3, 3)
    # p.merge(p1, 2)
    # p.append("c", 1)

    trees = {}
    # ### examples overlap/overtake
    # ### overtaking
    trees["P1"] = {0: {'p': 2, 'r': 4, 'children': [(1, 0)], 'parent': None},
                   1: {'event': 'a', 'parent': 0}}
    trees["P2"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
                   1: {'event': 'a', 'parent': 0}}
    trees["P3"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
                   1: {'p': 2, 'r': 4, 'children': [(2, 0)], 'parent': 0},
                   2: {'event': 'a', 'parent': 1}}
    trees["P4"] = {0: {'p': 2, 'r': 4, 'children': [(1, 0)], 'parent': None},
                   1: {'p': 13, 'r': 3, 'children': [(2, 0)], 'parent': 0},
                   2: {'event': 'a', 'parent': 1}}
    trees["P2b"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
                    1: {'event': 'b', 'parent': 0}}
    trees["P2c"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
                    1: {'event': 'c', 'parent': 0}}
    trees["P5"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0), (2, 3), (3, 1)], 'parent': None},
                   1: {'event': 'b', 'parent': 0},
                   2: {'event': 'a', 'parent': 0},
                   3: {'event': 'c', 'parent': 0}}

    # #####################################################
    # ### NESTING TWO CYCLES OVER THE SAME EVENT
    # #####################################################
    # #### simple example with one event
    # tmpOccs = {"a":12}

    # ### WITHOUT ERRORS
    # po = numpy.array([0, 2, 4, 6, 13, 15, 17, 19, 26, 28, 30, 32])#+2
    # ### WITH ERRORS
    # o = numpy.array([0, 3, 5, 6, 11, 13, 18, 19, 24, 27, 30, 31])+2
    # #o = numpy.array([2, 5, 7, 8, 13, 15, 20, 21, 26, 29, 32, 33])
    # collections = []
    # collections.append([("P1", o[i*4], numpy.diff(o[i*4:(i+1)*4])-trees["P1"][0]["p"]) for i in range(3)])
    # collections.append([("P2", o[i], numpy.diff(o[i::4])-trees["P2"][0]["p"]) for i in range(4)])

    # E = []
    # for i in range(len(collections[0])):
    #     if i > 0:
    #         E.append(collections[1][0][-1][i-1])
    #     E.extend(collections[0][i][-1])
    # collections.append([("P3", o[0], E)])

    # E = []
    # for i in range(len(collections[1])):
    #     if i > 0:
    #         E.append(collections[0][0][-1][i-1])
    #     E.extend(collections[1][i][-1])
    # collections.append([("P4", o[0], E)])
    # #####################################################

    #####################################################
    # CONCATENATING THREE CYCLES OVER DIFFERENT EVENTS
    #####################################################
    # simple example with one event
    tmpOccs = {"a": 3, "b": 3, "c": 3}

    # WITHOUT ERRORS
    po = numpy.array([0, 3, 4, 13, 16, 17, 26, 29, 30])  # +2
    # ### WITH ERRORS
    # o = numpy.array([0, 3, 5, 11, 13, 17, 24, 28, 29])+2
    o = numpy.array([0, 3, 5, 11, 16, 19, 24, 28, 29]) + 2
    ds = [v[1] for v in trees["P5"][0]["children"]]

    collections = []
    cpp = ["P2b", "P2", "P2c"]
    collections.append([(p, o[i], numpy.diff(o[i::len(cpp)]) -
                         trees["P2"][0]["p"]) for (i, p) in enumerate(cpp)])

    E = []
    for i in range(len(o)):
        if i > 0:
            if i % len(cpp) == 0:
                E.append((o[i] - o[i - len(cpp)]) - trees["P2"][0]["p"])
            else:
                E.append((o[i] - o[i - 1]) - ds[i % len(cpp)])
    collections.append([("P5", o[0], E)])
    #####################################################

    # RUN
    nbOccs, orgFreqs, adjFreqs, blck_delim = makeOccsAndFreqs(tmpOccs)
    print("ADJ_CL", [(k, numpy.log2(v)) for (k, v) in adjFreqs.items()])
    # t_end = numpy.max(o)
    # t_start = numpy.min(o)
    t_end = 34
    t_start = 0
    deltaT = t_end - t_start

    print("Sequence:", o, "\t", po)
    print("t_start=%d t_end=%d deltaT=%d" % (t_start, t_end, deltaT))

    for ci, col in enumerate(collections):
        print("==================")
        ccl = 0
        for pi, pat in enumerate(col):
            p = Pattern(trees[pat[0]])
            print("------------------")
            print("Pattern: Q_%d-%d\n" % (ci + 1, pi + 1), p)

            t0 = pat[1]
            occsStar = p.getOccsStar()
            oids = [o[-1] for o in occsStar]

            E = pat[2]
            Ed = p.getEDict(occsStar, E)
            print("Starting point:", t0)
            print("Corrections:", E, "\t", Ed)
            occs = p.getOccs(occsStar, t0, Ed)
            print("Occurrences:", sorted(occs), "\t", occs)
            data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
                            "nbOccs": nbOccs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}

            # print("------------------")
            # print("Pattern:\n", p)
            # print("Events:", p.getEventsList())
            # print("occs (%d) e,t:" % len(occs), [bo[:2] for bo in occs])

            # interleaved = {}
            # print("Time spanned:", p.timeSpanned(interleaved))
            # print("Interleaved:", p.isInterleaved())
            cl = p.codeLength(t0, E, data_details)
            print("Code length: %.3f" % cl)
            ccl += cl
        print("Collection %d code length: %.3f" % (ci, ccl))
    exit()

    # #### more examples
    # tmpOccs = {"a":50, "b": 100, "c": 50, "d": 40}
    # nbOccs, adjFreqs, orgFreqs, blck_delim = makeOccsAndFreqs(tmpOccs)

    # trees = []
    # # ### examples overlap/overtake
    # # ### overtaking
    # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0)], 'parent': None},
    #         1: {'event': 'c', 'parent': 0}})

    # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 5)], 'parent': None},
    # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
    # #         2: {'event': 'a', 'parent': 0},
    # #         3: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1}})
    # # ### overlaps, not overtaking
    # # trees.append({0: {'p': 8, 'r': 2, 'children': [(1, 0), (2, 8)], 'parent': None},
    # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
    # #         2: {'event': 'a', 'parent': 0},
    # #         3: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1}})
    # # ### no overlaps
    # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 9)], 'parent': None},
    # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
    # #         2: {'event': 'a', 'parent': 0},
    # #         3: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1}})

    # # ### complex pattern
    # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 5)], 'parent': None},
    # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2), (5, 1)], 'parent': 0},
    # #         3: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1},
    # #         5: {'event': 'd', 'parent': 1},
    # #         2: {'p': 4, 'r': 3, 'children': [(6, 0)], 'parent': 0},
    # #         6: {'p': 1, 'r': 2, 'children': [(7, 0)], 'parent': 2},
    # #         7: {'event': 'a', 'parent': 6}})

    # # ### complex pattern P8
    # # trees.append({0: {'p': 5, 'r': 2, 'children': [(1, 0)], 'parent': None},
    # #         1: {'p': 10, 'r': 3, 'children': [(2, 0), (3, 3), (4, 1)], 'parent': 0},
    # #         2: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1},
    # #         3: {'p': 1, 'r': 4, 'children': [(5, 0)], 'parent': 1},
    # #         5: {'event': 'a', 'parent': 3}})

    # ### variation on P8, interleaving, no overlaps
    # # trees.append({0: {'p': 24, 'r': 2, 'children': [(1, 0)], 'parent': None},
    # #         1: {'p': 8, 'r': 3, 'children': [(2, 0), (3, 3), (4, 1)], 'parent': 0},
    # #         2: {'event': 'b', 'parent': 1},
    # #         4: {'event': 'c', 'parent': 1},
    # #         3: {'p': 2, 'r': 4, 'children': [(5, 0)], 'parent': 1},
    # #         5: {'event': 'a', 'parent': 3}})

    # ### variation on P8, no interleaving, no overlaps
    # trees.append({0: {'p': 33, 'r': 2, 'children': [(1, 0)], 'parent': None},
    #         1: {'p': 10, 'r': 3, 'children': [(2, 0), (3, 3), (4, 5)], 'parent': 0},
    #         2: {'event': 'b', 'parent': 1},
    #         4: {'event': 'c', 'parent': 1},
    #         3: {'p': 1, 'r': 4, 'children': [(5, 0)], 'parent': 1},
    #         5: {'event': 'a', 'parent': 3}})

    # # ### examples nested cycles
    # # ### longest period first: no interleaving
    # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0)], 'parent': None},
    # #         1: {'p': 3, 'r': 3, 'children': [(2, 0)], 'parent': 0},
    # #         2: {'event': 'a', 'parent': 0}})
    # # ### short period first: overtaking itself
    # # trees.append({0: {'p': 3, 'r': 3, 'children': [(1, 0)], 'parent': None},
    # #         1: {'p': 10, 'r': 2, 'children': [(2, 0)], 'parent': 0},
    # #         2: {'event': 'a', 'parent': 0}})

    # noise = [ 1,  1, -1, -2,  1,  0,  1,  1,  1,  0, -2, -2,  0,  1,  1,  0,  0,
    #          -1, -2,  1, -2,  1, -2,  0,  0, -1, -2, -1,  1, -1,  0, -2,  1, -2,
    #          -2,  1,  0, -1, -1,  1,  0, -1,  1,  1,  1,  1, -2,  0,  1,  1,  0]
    # print("Noise:", noise)

    # for ti, tree in enumerate(trees):
    #     p = Pattern(tree)
    #     print("------------------")
    #     print("Pattern:\n", p)

    #     t0 = 10
    #     occsStar = p.getOccsStar()
    #     oids = [o[-1] for o in occsStar]

    #     if len(occsStar) < len(noise):
    #         E = noise[:len(occsStar)-1]
    #     else:
    #         E = numpy.random.randint(-2,2, size=len(occsStar)-1)
    #     E = []

    #     Ed = p.getEDict(occsStar, E)
    #     occs = p.getOccs(occsStar, t0, Ed)
    #     # occsRef = p.getOccsByRefs(occsStar, t0, Ed)

    #     # print(occs)
    #     # print(occsRef)
    #     # occsD = dict(zip(*[oids, occs]))
    #     # rEd, rt0 = p.computeEDict(occsD)

    #     t_end = numpy.max(occs)
    #     t_start = numpy.min(occs)
    #     deltaT = t_end - t_start
    #     data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
    #                      "nbOccs": nbOccs, "orgFreqs": orgFreqs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}

    #     # print("------------------")
    #     # print("Pattern:\n", p)
    #     print("Events:", p.getEventsList())
    #     print("depth=%d width=%d alphabet=%s" % (p.getDepth(), p.getWidth(), p.getAlphabet()))
    #     print(len([bo[0] for bo in occsStar]), len(set([bo[0] for bo in occsStar])))
    #     print("occs (%d) e,t:" % len(occs), [bo[:2] for bo in occsStar])
    #     print("(%s)" % "),$ $(".join(["%d, %s" % tuple(bo[:2]) for bo in occsStar]))
    #     print("{%s}" % ",".join(["%d/%s" % tuple(bo[:2]) for bo in occsStar]))

    #     # interleaved = {}
    #     # print("Time spanned:", p.timeSpanned(interleaved))
    #     # print("Interleaved:", p.isInterleaved())
    #     print("Code length:", p.codeLength(t0, E, data_details))
