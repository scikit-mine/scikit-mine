import itertools

import numpy

# prop_list from Candidate
# list of the names of the properties of the pattern
prop_list = ["t0i", "p0", "r0", "offset", "cumEi", "new", "cid"]
# Dictionary mapping property names to their index in `prop_list`.
prop_map = dict([(v, k) for k, v in enumerate(prop_list)])

# FIXME : to be explained
# OPT_TO = False
OPT_TO = True
OPT_EVFR = True  # False
# ADJFR_MED = False
# PROPS_MAX_OFFSET = 3
# PROPS_MIN_R = 3
PROPS_MAX_OFFSET = -2
PROPS_MIN_R: int = 0


def _getChained(listsd, keys=None):
    if keys is None:
        keys = list(listsd.keys())
    return itertools.chain(*[listsd.get(k, []) for k in keys])


def computePeriodDiffs(diffs):
    """
    From a list of timestamp diffs, computes the period of a pattern.

    This method calculates the median of the differences and then takes the floor of this value. The floor of the scalar
    x is the largest integer i, such that i <= x.

    Parameters
    ----------
    diffs : list
        List of timestamp differences

    Returns
    -------
    int
        The pattern period
    """
    return int(numpy.floor(numpy.median(diffs)))


def computePeriod(occs, sort=False):
    """
    Calculates the period of a pattern from its timestamps

    Parameters
    ----------
    occs : list
        The list of timestamps

    sort : bool, default=False
        Whether or not to sort the timestamps of occs

    Returns
    -------
    int
        The pattern period
    """
    if sort:
        occs = sorted(occs)
    return computePeriodDiffs(numpy.diff(occs))


def computeE(occs, p0, sort=False):
    """
    Computes the list of shifts derivations from a candidate from its occurrences and its period

    Parameters
    ----------
    occs : list
        List of occurences/timestamps of a candidate
    p0 : int
        Period of the candidate
    sort : bool, default=False
        Sort or not the occs list

    Returns
    -------
    The list of the cycle shift derivations

    """
    if sort:
        occs = sorted(occs)
    return [(occs[i] - occs[i - 1]) - p0 for i in range(1, len(occs))]


def cost_triple(data_details, alpha, dp, deltaE):
    """
    FIXME : to be explained
    Parameters
    ----------
    data_details
    alpha
    dp
    deltaE

    Returns
    -------

    """
    if (data_details["deltaT"] - deltaE + 1) / 2. - dp < 0:
        print("!!!---- Problem delta", data_details["deltaT"], deltaE, dp)

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
    """
    Compute L(E)
    .. math:: L(E) = 2*|E| + \sum_{e \in E}|e| \text{where} |E|=|occs*(P)|-1

    Parameters
    ----------
    occs : list
        List of all occurrences of a candidate

    cp : int
        Candidate period

    Returns
    -------
    int
        L(E)
    """
    return numpy.sum([2 + numpy.abs((occs[i] - occs[i - 1]) - cp) for i in range(1, len(occs))])


def computeLengthCycle(data_details, cycle, print_dets=False, no_err=False):
    """
        Compute the length of a cycle
        L(P) = L(A) + L(R) + L(p) + L(D) + L(to) + L(E)

    Parameters
    ----------
    data_details
    cycle
    print_dets
    no_err

    Returns
    -------
    int
        L(P)

    """
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
            # to compute noE
            # rr cost:
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
    """
        Compute the codelength of the residuals
    Parameters
    ----------
    data_details
    residual : dict
        Contains the following keys:
        * alpha: the event
        * occs: all occurences where alpha is not covered by the cycle

    Returns
    -------
    float
        Codelength of the residuals
    """
    residual["cp"] = -1
    return len(residual["occs"]) * cost_one(data_details, residual["alpha"])


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
        A float representing the size encoded of the block delimiter of the sequence.
    """
    return makeOccsAndFreqsThird(tmpOccs)


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
    """
    Takes as input a string where each intrinsic element is separated by SUB_SEP and these elements are separated from
    the others by SUB_SEP. The function returns a list of lists.

    Parameters
    ----------
    key : str

    Returns
    -------
        list of list
    """
    try:
        return [list(map(int, b.split(SUB_SEP))) for b in key.split(SUP_SEP)]
    except:
        return []


def l_to_key(l):
    """
    Takes as input a list of pairs and returns a string where the elements of each tuple are separated
    by SUB_SEP and each tuple by SUP_SEP.

    Parameters
    ----------
    l : list
        The list containing tuples of 2 items

    Returns
    -------
        str
    """
    return SUP_SEP.join([("%d" + SUB_SEP + "%d") % tuple(pf) for pf in l])


def l_to_br(l):
    """
     Convert a list of pairs of integers to a string of the form "B<i1,...,in><j1,...,jn>",
     where n is the length of the input list and each pair of integers (i,j) in the input
     list is represented as "<i+1,j+1>". The output string starts with the letter "B".

     Parameters
     ----------
     l : list of pairs of integers
         The input list to convert to the string format. Each pair of integers (i,j) in the
         input list should be represented as a list [i, j].

     Returns
     -------
     str
         A string of the form "B<i1,...,in><j1,...,jn>", where n is the length of the input
         list and each pair of integers (i,j) in the input list is represented as "<i+1,j+1>".
    """
    return "B" + ",".join(["%d" % (int(pf[0]) + 1) for pf in l]) + "<" + ",".join(
        ["%d" % (int(pf[1]) + 1) for pf in l]) + ">"


def key_to_br(key):
    l = key_to_l(key)
    return l_to_br(l)


def propCmp(props, pid):
    """
    Get the t0i, p0, and r0 properties of a given pattern id.

    Parameters
    ----------
    props : list or numpy ndarray
    pid : int
        The ID of the pattern whose properties are being requested.

    Returns
    -------
    tuple of floats
        A tuple containing the t0i, p0, and r0 properties of the requested particle.
    """
    if type(props) is list:
        return (props[pid][prop_map["t0i"]],
                props[pid][prop_map["p0"]],
                props[pid][prop_map["r0"]])
    else:
        return (props[pid, prop_map["t0i"]],
                props[pid, prop_map["p0"]],
                props[pid, prop_map["r0"]])


def sortPids(props, pids=None):
    """
    Sorts a list of pattern ids based on their property values. The sorting is done in tuple order (result of propCmp)
    and based on the values of the tuple elements.

    Parameters
    ----------
    props : list or numpy.ndarray
    pids : list or None, optional
        A list of pattern IDs to sort.
        If None, all property IDs will be sorted.
        Default is None.

    Returns
    -------
    list
        A list of pattern IDs sorted based on their property values.
    """
    if pids is None:
        pids = range(len(props))
    return sorted(pids, key=lambda x: propCmp(props, x))


def mergeSortedPids(props, pidsA, pidsB):
    """
        Merges two lists of pids (pidsA and pidsB) and sorts them first by t0 then p0 and r0 (from props).
    Parameters
    ----------

    props : ndarray
    pidsA : list
    pidsB : list

    Returns
    -------
        list
    Returns a merged and sorted list of pids
    """
    i = 0
    while i < len(pidsA) and len(pidsB) > 0:
        if propCmp(props, pidsA[i]) > propCmp(props, pidsB[0]):
            pidsA.insert(i, pidsB.pop(0))
        i += 1
    if len(pidsB) > 0:
        pidsA.extend(pidsB)
    return pidsA
