import json

import numpy as np

from .class_patterns import cost_one


def _replace_tuple_in_list(l, i):
    """
    replace int64 into int in a tuple
    """
    if i < len(l):
        l[i] = tuple(int(v) if isinstance(v, np.int64) else v for v in l[i])
    return l


def _replace_list_in_list(l, i):
    """
    replace int64 into int in a list
    """
    if i < len(l):
        l[i] = [int(v) if isinstance(v, np.int64) else v for v in l[i]]
    return l


def _change_int64_toint(obj):
    """
    convert a int64 into int in a nested dict with tuple 
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                _change_int64_toint(value)
            elif isinstance(value, list):
                for it, val_ in enumerate(value):
                    if isinstance(val_, tuple):
                        value = _replace_tuple_in_list(value, it)
                    if isinstance(val_, list):
                        value = _replace_list_in_list(value, it)
                _change_int64_toint(value)
            elif isinstance(value, tuple):
                _change_int64_toint(value)
            elif isinstance(value, np.int64):
                value = int(value)
                obj[key] = value
    return obj


class PatternCollection(object):
    """
    A class representing a collection of patterns.

    Parameters
    ----------
    patterns : list of tuple, optional (default=[])
        List of patterns, where each pattern is represented as a tuple of three elements:
        1) the pattern object;
        2) an integer representing the starting time of the pattern occurrences;
        3) a list of integers representing the shift corrections.

    Attributes
    ----------
    patterns : list
        A list of pattern tuples.

    occ_lists : list of list, optional (default=None)
        List of lists of integers representing the positions of the pattern occurrences in the
        input data sequence.
    """

    def __init__(self, patterns=[]):
        self.patterns = patterns
        self.occ_lists = None

    def __len__(self):
        return len(self.patterns)

    def nbPatternsByType(self):
        """
        Return a dictionary that counts the number of patterns by type (simple, nested, complex, other).

        Returns
        -------
        dict
            A dictionary where the keys are the pattern types and the values are the number of patterns of that type.
        """
        nbs = {}
        for (P, t0, E) in self.patterns:
            tstr = P.getTypeStr()
            nbs[tstr] = nbs.get(tstr, 0) + 1
        return nbs

    def getPatterns(self):
        return self.patterns

    def getUncoveredOccs(self, data_seq):
        """
        Get the occurrences that are not covered by the current cycle

        Parameters
        ----------
        data_seq : DataSequence
            The sequence with all occurrences

        Returns
        -------
        set
            A set of tuples each containing a timestamp and the associated event (integer)
        """
        return set(data_seq.getSequence()).difference(*self.getOccLists())

    def getNbUncoveredOccsByEv(self, data_seq):
        """
        Get a dictionary with the number of uncovered occurences for each event

        Parameters
        ----------
        data_seq : DataSequence
            The sequence with all occurences

        Returns
        -------
        dict
            A dictionary with the number of uncovered occurrences for each event

        """
        nbs = {}
        for (t, ev) in self.getUncoveredOccs(data_seq):
            nbs[ev] = nbs.get(ev, 0) + 1
        return nbs

    def getOccLists(self):
        """
        Returns a list of lists, where each sublist contains the occurrences of a given pattern

        Returns
        -------
        List[List[Tuple[int, int]]]
            A list of lists, where each sublist contains the occurrences of a given pattern.
        """
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
                    map_ev=map_ev, leaves_first=True), clp, np.sum(np.abs(E)), len(ocls[pi]), len(set(ocls[pi])),
                    p.getTypeStr())
            cl += clp
        return str_out, cl

    def strDetailed(self, data_seq, print_simple=True):
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
                    map_ev=map_ev, leaves_first=True), clp, np.sum(np.abs(E)), len(ocls[pi]), len(set(ocls[pi])),
                    p.getTypeStr())

            pattern_tree = p.nodes
            pattern_tree["next_id"] = p.next_id
            pattern_tree["t0"] = int(t0)
            pattern_tree["E"] = [int(e) for e in E]
            pattern_tree = json.dumps(_change_int64_toint(pattern_tree))

            dict_pattern["t0"] = t0
            dict_pattern["pattern_json_tree"] = pattern_tree
            dict_pattern["pattern"] = p.__str__(
                map_ev=map_ev, leaves_first=True)
            dict_pattern["repetition_major"] = p.pattMajorKey_list()[0]
            dict_pattern["period_major"] = p.pattMajorKey_list()[1]
            # dict_pattern["cost"] = clp
            dict_pattern["E"] = E

            patterns_list_of_dict.append(dict_pattern)

            cl += clp

        return patterns_list_of_dict, cl

    def output_detailed(self, data_seq, print_simple=True, auto_time_scale_factor=1):
        nbs = self.nbPatternsByType()
        data_details = data_seq.getDetails()

        patterns_list_of_dict, cl = self.output_pattern_list_and_cost(
            data_seq, print_simple, auto_time_scale_factor=auto_time_scale_factor)

        global_stat_dict = {"Total patterns nb": len(self)}
        for (k, v) in sorted(nbs.items(), key=lambda x: -x[1]):
            global_stat_dict[k] = v
        nbU = self.getNbUncoveredOccsByEv(data_seq)
        clR = 0
        nbR = 0
        for (ev, nb) in nbU.items():
            nbR += nb
            clR += nb * cost_one(data_details, ev)

        global_stat_dict[f"Code length patterns"] = cl
        global_stat_dict[f"Code length residuals"] = clR
        global_stat_dict["Total residual nb"] = nbR

        cl += clR
        clRonly = data_seq.codeLengthResiduals()
        if clRonly > 0:
            global_stat_dict["Code length total"] = cl
            global_stat_dict["Total compression ratio"] = 100 * cl / clRonly
            global_stat_dict["Code length residuals only"] = clRonly

        return global_stat_dict, patterns_list_of_dict
