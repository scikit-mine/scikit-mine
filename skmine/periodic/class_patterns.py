import itertools
import pdb

import numpy

# prop_list from Candidate
# list of the names of the properties of the pattern
prop_list = ["t0i", "p0", "r0", "offset", "cumEi", "new", "cid"]
# Dictionary mapping property names to their index
prop_map = dict([(v, k) for k, v in enumerate(prop_list)])
# in `prop_list`.

# OPT_TO = False
OPT_TO = True
OPT_EVFR = True  # False
ADJFR_MED = False
# PROPS_MAX_OFFSET = 3
# PROPS_MIN_R = 3
PROPS_MAX_OFFSET = -2
PROPS_MIN_R: int = 0


def _replace_tuple_in_list(l, i):
    """
    replace int64 into int in a tuple
    """
    if i < len(l):
        l[i] = tuple(int(v) if isinstance(v, numpy.int64) else v for v in l[i])
    return l


def _replace_list_in_list(l, i):
    """
    replace int64 into int in a list
    """
    if i < len(l):
        l[i] = [int(v) if isinstance(v, numpy.int64) else v for v in l[i]]
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
            elif isinstance(value, numpy.int64):
                value = int(value)
                obj[key] = value
    return obj


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
    """
    Compute the cost of an individual occurence o = (t, alpha) based on the data_details.
    L(o) = L(t) + L(alpha) = log(deltaS + 1) - log2(freq(alpha))

    Parameters
    ----------
    data_details : dict
        A dictionary containing the following keys:
        * nbOccs (dict) - a dictionary of itemsets and their frequencies
        * orgFreqs (dict) - a dictionary of itemsets and their original frequencies
        * adjFreqs (dict) - a dictionary of itemsets and their adjusted frequencies
        * deltaT (int) - the total number of transactions in the dataset
    alpha : int
        The event for which to compute the cost.

    Returns
    -------
    float
        The cost of the occurence.
    """
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


def propCmp(props, pid):
    if type(props) is list:
        return (props[pid][prop_map["t0i"]],
                props[pid][prop_map["p0"]],
                props[pid][prop_map["r0"]])
    else:
        return (props[pid, prop_map["t0i"]],
                props[pid, prop_map["p0"]],
                props[pid, prop_map["r0"]])


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


# if __name__ == "__main__":
#     # p = Pattern("a", 2, 3)
#     # p1 = Pattern("b", 3, 3)
#     # p.merge(p1, 2)
#     # p.append("c", 1)
#
#     trees = {}
#     # ### examples overlap/overtake
#     # ### overtaking
#     trees["P1"] = {0: {'p': 2, 'r': 4, 'children': [(1, 0)], 'parent': None},
#                    1: {'event': 'a', 'parent': 0}}
#     trees["P2"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
#                    1: {'event': 'a', 'parent': 0}}
#     trees["P3"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
#                    1: {'p': 2, 'r': 4, 'children': [(2, 0)], 'parent': 0},
#                    2: {'event': 'a', 'parent': 1}}
#     trees["P4"] = {0: {'p': 2, 'r': 4, 'children': [(1, 0)], 'parent': None},
#                    1: {'p': 13, 'r': 3, 'children': [(2, 0)], 'parent': 0},
#                    2: {'event': 'a', 'parent': 1}}
#     trees["P2b"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
#                     1: {'event': 'b', 'parent': 0}}
#     trees["P2c"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0)], 'parent': None},
#                     1: {'event': 'c', 'parent': 0}}
#     trees["P5"] = {0: {'p': 13, 'r': 3, 'children': [(1, 0), (2, 3), (3, 1)], 'parent': None},
#                    1: {'event': 'b', 'parent': 0},
#                    2: {'event': 'a', 'parent': 0},
#                    3: {'event': 'c', 'parent': 0}}
#
#     # #####################################################
#     # ### NESTING TWO CYCLES OVER THE SAME EVENT
#     # #####################################################
#     # #### simple example with one event
#     # tmpOccs = {"a":12}
#
#     # ### WITHOUT ERRORS
#     # po = numpy.array([0, 2, 4, 6, 13, 15, 17, 19, 26, 28, 30, 32])#+2
#     # ### WITH ERRORS
#     # o = numpy.array([0, 3, 5, 6, 11, 13, 18, 19, 24, 27, 30, 31])+2
#     # #o = numpy.array([2, 5, 7, 8, 13, 15, 20, 21, 26, 29, 32, 33])
#     # collections = []
#     # collections.append([("P1", o[i*4], numpy.diff(o[i*4:(i+1)*4])-trees["P1"][0]["p"]) for i in range(3)])
#     # collections.append([("P2", o[i], numpy.diff(o[i::4])-trees["P2"][0]["p"]) for i in range(4)])
#
#     # E = []
#     # for i in range(len(collections[0])):
#     #     if i > 0:
#     #         E.append(collections[1][0][-1][i-1])
#     #     E.extend(collections[0][i][-1])
#     # collections.append([("P3", o[0], E)])
#
#     # E = []
#     # for i in range(len(collections[1])):
#     #     if i > 0:
#     #         E.append(collections[0][0][-1][i-1])
#     #     E.extend(collections[1][i][-1])
#     # collections.append([("P4", o[0], E)])
#     # #####################################################
#
#     #####################################################
#     # CONCATENATING THREE CYCLES OVER DIFFERENT EVENTS
#     #####################################################
#     # simple example with one event
#     tmpOccs = {"a": 3, "b": 3, "c": 3}
#
#     # WITHOUT ERRORS
#     po = numpy.array([0, 3, 4, 13, 16, 17, 26, 29, 30])  # +2
#     # ### WITH ERRORS
#     # o = numpy.array([0, 3, 5, 11, 13, 17, 24, 28, 29])+2
#     o = numpy.array([0, 3, 5, 11, 16, 19, 24, 28, 29]) + 2
#     ds = [v[1] for v in trees["P5"][0]["children"]]
#
#     collections = []
#     cpp = ["P2b", "P2", "P2c"]
#     collections.append([(p, o[i], numpy.diff(o[i::len(cpp)]) -
#                          trees["P2"][0]["p"]) for (i, p) in enumerate(cpp)])
#
#     E = []
#     for i in range(len(o)):
#         if i > 0:
#             if i % len(cpp) == 0:
#                 E.append((o[i] - o[i - len(cpp)]) - trees["P2"][0]["p"])
#             else:
#                 E.append((o[i] - o[i - 1]) - ds[i % len(cpp)])
#     collections.append([("P5", o[0], E)])
#     #####################################################
#
#     # RUN
#     nbOccs, orgFreqs, adjFreqs, blck_delim = makeOccsAndFreqs(tmpOccs)
#     print("ADJ_CL", [(k, numpy.log2(v)) for (k, v) in adjFreqs.items()])
#     # t_end = numpy.max(o)
#     # t_start = numpy.min(o)
#     t_end = 34
#     t_start = 0
#     deltaT = t_end - t_start
#
#     print("Sequence:", o, "\t", po)
#     print("t_start=%d t_end=%d deltaT=%d" % (t_start, t_end, deltaT))
#
#     for ci, col in enumerate(collections):
#         print("==================")
#         ccl = 0
#         for pi, pat in enumerate(col):
#             p = Pattern(trees[pat[0]])
#             print("------------------")
#             print("Pattern: Q_%d-%d\n" % (ci + 1, pi + 1), p)
#
#             t0 = pat[1]
#             occsStar = p.getOccsStar()
#             oids = [o[-1] for o in occsStar]
#
#             E = pat[2]
#             Ed = p.getEDict(occsStar, E)
#             print("Starting point:", t0)
#             print("Corrections:", E, "\t", Ed)
#             occs = p.getOccs(occsStar, t0, Ed)
#             print("Occurrences:", sorted(occs), "\t", occs)
#             data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
#                             "nbOccs": nbOccs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}
#
#             # print("------------------")
#             # print("Pattern:\n", p)
#             # print("Events:", p.getEventsList())
#             # print("occs (%d) e,t:" % len(occs), [bo[:2] for bo in occs])
#
#             # interleaved = {}
#             # print("Time spanned:", p.timeSpanned(interleaved))
#             # print("Interleaved:", p.isInterleaved())
#             cl = p.codeLength(t0, E, data_details)
#             print("Code length: %.3f" % cl)
#             ccl += cl
#         print("Collection %d code length: %.3f" % (ci, ccl))
#     exit()
#
#     # #### more examples
#     # tmpOccs = {"a":50, "b": 100, "c": 50, "d": 40}
#     # nbOccs, adjFreqs, orgFreqs, blck_delim = makeOccsAndFreqs(tmpOccs)
#
#     # trees = []
#     # # ### examples overlap/overtake
#     # # ### overtaking
#     # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0)], 'parent': None},
#     #         1: {'event': 'c', 'parent': 0}})
#
#     # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 5)], 'parent': None},
#     # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
#     # #         2: {'event': 'a', 'parent': 0},
#     # #         3: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1}})
#     # # ### overlaps, not overtaking
#     # # trees.append({0: {'p': 8, 'r': 2, 'children': [(1, 0), (2, 8)], 'parent': None},
#     # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
#     # #         2: {'event': 'a', 'parent': 0},
#     # #         3: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1}})
#     # # ### no overlaps
#     # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 9)], 'parent': None},
#     # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2)], 'parent': 0},
#     # #         2: {'event': 'a', 'parent': 0},
#     # #         3: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1}})
#
#     # # ### complex pattern
#     # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0), (2, 5)], 'parent': None},
#     # #         1: {'p': 3, 'r': 3, 'children': [(3, 0), (4, 2), (5, 1)], 'parent': 0},
#     # #         3: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1},
#     # #         5: {'event': 'd', 'parent': 1},
#     # #         2: {'p': 4, 'r': 3, 'children': [(6, 0)], 'parent': 0},
#     # #         6: {'p': 1, 'r': 2, 'children': [(7, 0)], 'parent': 2},
#     # #         7: {'event': 'a', 'parent': 6}})
#
#     # # ### complex pattern P8
#     # # trees.append({0: {'p': 5, 'r': 2, 'children': [(1, 0)], 'parent': None},
#     # #         1: {'p': 10, 'r': 3, 'children': [(2, 0), (3, 3), (4, 1)], 'parent': 0},
#     # #         2: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1},
#     # #         3: {'p': 1, 'r': 4, 'children': [(5, 0)], 'parent': 1},
#     # #         5: {'event': 'a', 'parent': 3}})
#
#     # ### variation on P8, interleaving, no overlaps
#     # # trees.append({0: {'p': 24, 'r': 2, 'children': [(1, 0)], 'parent': None},
#     # #         1: {'p': 8, 'r': 3, 'children': [(2, 0), (3, 3), (4, 1)], 'parent': 0},
#     # #         2: {'event': 'b', 'parent': 1},
#     # #         4: {'event': 'c', 'parent': 1},
#     # #         3: {'p': 2, 'r': 4, 'children': [(5, 0)], 'parent': 1},
#     # #         5: {'event': 'a', 'parent': 3}})
#
#     # ### variation on P8, no interleaving, no overlaps
#     # trees.append({0: {'p': 33, 'r': 2, 'children': [(1, 0)], 'parent': None},
#     #         1: {'p': 10, 'r': 3, 'children': [(2, 0), (3, 3), (4, 5)], 'parent': 0},
#     #         2: {'event': 'b', 'parent': 1},
#     #         4: {'event': 'c', 'parent': 1},
#     #         3: {'p': 1, 'r': 4, 'children': [(5, 0)], 'parent': 1},
#     #         5: {'event': 'a', 'parent': 3}})
#
#     # # ### examples nested cycles
#     # # ### longest period first: no interleaving
#     # # trees.append({0: {'p': 10, 'r': 2, 'children': [(1, 0)], 'parent': None},
#     # #         1: {'p': 3, 'r': 3, 'children': [(2, 0)], 'parent': 0},
#     # #         2: {'event': 'a', 'parent': 0}})
#     # # ### short period first: overtaking itself
#     # # trees.append({0: {'p': 3, 'r': 3, 'children': [(1, 0)], 'parent': None},
#     # #         1: {'p': 10, 'r': 2, 'children': [(2, 0)], 'parent': 0},
#     # #         2: {'event': 'a', 'parent': 0}})
#
#     # noise = [ 1,  1, -1, -2,  1,  0,  1,  1,  1,  0, -2, -2,  0,  1,  1,  0,  0,
#     #          -1, -2,  1, -2,  1, -2,  0,  0, -1, -2, -1,  1, -1,  0, -2,  1, -2,
#     #          -2,  1,  0, -1, -1,  1,  0, -1,  1,  1,  1,  1, -2,  0,  1,  1,  0]
#     # print("Noise:", noise)
#
#     # for ti, tree in enumerate(trees):
#     #     p = Pattern(tree)
#     #     print("------------------")
#     #     print("Pattern:\n", p)
#
#     #     t0 = 10
#     #     occsStar = p.getOccsStar()
#     #     oids = [o[-1] for o in occsStar]
#
#     #     if len(occsStar) < len(noise):
#     #         E = noise[:len(occsStar)-1]
#     #     else:
#     #         E = numpy.random.randint(-2,2, size=len(occsStar)-1)
#     #     E = []
#
#     #     Ed = p.getEDict(occsStar, E)
#     #     occs = p.getOccs(occsStar, t0, Ed)
#     #     # occsRef = p.getOccsByRefs(occsStar, t0, Ed)
#
#     #     # print(occs)
#     #     # print(occsRef)
#     #     # occsD = dict(zip(*[oids, occs]))
#     #     # rEd, rt0 = p.computeEDict(occsD)
#
#     #     t_end = numpy.max(occs)
#     #     t_start = numpy.min(occs)
#     #     deltaT = t_end - t_start
#     #     data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
#     #                      "nbOccs": nbOccs, "orgFreqs": orgFreqs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}
#
#     #     # print("------------------")
#     #     # print("Pattern:\n", p)
#     #     print("Events:", p.getEventsList())
#     #     print("depth=%d width=%d alphabet=%s" % (p.getDepth(), p.getWidth(), p.getAlphabet()))
#     #     print(len([bo[0] for bo in occsStar]), len(set([bo[0] for bo in occsStar])))
#     #     print("occs (%d) e,t:" % len(occs), [bo[:2] for bo in occsStar])
#     #     print("(%s)" % "),$ $(".join(["%d, %s" % tuple(bo[:2]) for bo in occsStar]))
#     #     print("{%s}" % ",".join(["%d/%s" % tuple(bo[:2]) for bo in occsStar]))
#
#     #     # interleaved = {}
#     #     # print("Time spanned:", p.timeSpanned(interleaved))
#     #     # print("Interleaved:", p.isInterleaved())
#     #     print("Code length:", p.codeLength(t0, E, data_details))
