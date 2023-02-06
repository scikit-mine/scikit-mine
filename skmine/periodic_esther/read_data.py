import numpy
import re

group_patt = "GROUP_%d"
group_syms = "^*$"


def readSequence(parameters):
    sequences = dict([(group_patt % pi, []) for pi, p in enumerate(
        parameters.get("events", [])) if all([sym in p for sym in group_syms])])

    print("group_patt", group_patt)
    print("group_syms", group_syms)
    print("sequences", sequences)
    li = 0
    if "filename" in parameters:
        with open(parameters["filename"]) as fp:
            for line in fp:
                if re.match("#", line):
                    continue
                parts = line.strip().split(parameters.get("SEP", "\t"))
                if not re.match(" *%", line) and len(parts) == 2:
                    li += 1
                    for pi, p in enumerate(parameters.get("events", [None])):
                        if p is None or p == "*" or (p != "*" and re.search(p, parts[1])):
                            if p is not None and all([sym in p for sym in group_syms]):
                                k = group_patt % pi
                            else:
                                k = parts[1]
                                if k not in sequences:
                                    sequences[k] = []

                            if parameters.get("timestamp", True):
                                sequences[k].append(int(parts[0]))
                            else:
                                sequences[k].append(li)

    ss = {}
    for sk, s in sequences.items():
        if parameters.get("min_len", 0) < len(s) < parameters.get("max_len", len(s) + 1):
            ss[sk] = numpy.array(sorted(set(s)))
    return ss


def readSequenceSacha(parameters):
    # granularity, only with timestamps
    print(parameters)
    withI = float(parameters.get("I", False))
    granularity = float(parameters.get("granularity", 1))
    drop_event_codes = parameters.get("drop_event_codes", None)

    print("parameters", parameters)

    if "events" in parameters:
        sequences = dict([(pi, [])
                         for pi, p in enumerate(parameters["events"])])
    else:
        sequences = {}

    li = 0
    if "filename" in parameters:
        with open(parameters["filename"]) as fp:
            for line in fp:
                if re.match("#", line):
                    continue
                parts = line.strip().split(" ")
                if not re.match(" *%", line) and len(parts) == 7:
                    li += 1
                    k = parts[0]
                    event_code = int(k)
                    if drop_event_codes is None or event_code not in drop_event_codes:
                        if parameters.get("timestamp", True):
                            t_start, t_end, t_delta = (
                                int(parts[1]), int(parts[2]), int(parts[3]))
                            if withI and t_delta < granularity:
                                kk = "%s_I" % k
                                if kk not in sequences:
                                    sequences[kk] = []
                                sequences[kk].append(
                                    int(int(t_start)/granularity))
                            else:
                                for (suff, tt) in [("S", t_start), ("E", t_end)]:
                                    kk = "%s_%s" % (k, suff)
                                    if kk not in sequences:
                                        sequences[kk] = []
                                    sequences[kk].append(
                                        int(int(tt)/granularity))
                        else:
                            if k not in sequences:
                                sequences[k] = []
                            sequences[k].append(li)

    ss = {}
    for sk, s in sequences.items():
        if parameters.get("min_len", 0) < len(s) < parameters.get("max_len", len(s) + 1):
            ss[sk] = numpy.array(sorted(set(s)))
    return ss
