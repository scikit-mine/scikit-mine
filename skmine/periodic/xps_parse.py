import re
import datetime
import glob
import sys
import numpy
import pdb

MIN_OCC = 10

SERIES = "vX"
if len(sys.argv) > 1:
    SERIES = sys.argv[1]

XPS_REP = "../xps/"
XPS_SUB = "runs/"
LOG_MATCH = "*%s_log.txt*" % SERIES
PATT_MATCH = "_patt*.txt"
OUT_FILE = "%s.csv" % SERIES

INTERMS = [('F', 'patts', 'Final'), ('S', 'patts-simple', 'Simple'),
           ('V', 'patts-simple+V', 'Simple+V'), ('H', 'patts-simple+H', 'Simple+H'),
           ('V+H', 'patts-simple+V+H', 'Simple+V+H')]
MAIN_STATS = ['compressed_cl', 'original_cl', 'prc_cl', 'total', 'patt_cl', 'nb_patts', 'residuals_cl', 'nb_residuals',
              'nb_simple', 'nb_concat', 'nb_nested', 'nb_other']
SEC_STATS = ['nb_cands', 'time']
MAIN_TIMES = [('cycles', 'simple cycle mining'),
              ('combine', 'Combinations'), ('mining', 'Mining_only')]


def timedeltastr_to_num(dT):
    tmp = re.match(
        "(?P<days>\d+ days?, )?(?P<hours>\d?\d):(?P<mins>\d\d):(?P<secs>\d\d)\.(?P<ms>\d+)$", dT)
    if tmp is not None:
        if tmp.group("days") is None:
            days = 0
        else:
            days = int(tmp.group("days"))
        dd = datetime.timedelta(days=days, hours=int(tmp.group("hours")), minutes=int(tmp.group("mins")),
                                seconds=int(tmp.group("secs")), microseconds=int(tmp.group("ms")))
        return dd.total_seconds()
    return -1.


def parse_log_times(fn):
    time_lines = []
    inter_lines = []
    more_stats = {"evs_O": []}
    with open(fn) as fp:
        for line in fp:
            if re.match("\[TIME\]", line):
                time_lines.append(line)
            elif re.match("\[INTER\]", line):
                inter_lines.append(line)
            elif re.search("selection \([0-9]* candidates\)", line):
                inter_lines.append(line)
            else:
                tmp = re.match(
                    "\t\S+ \[\d+\] \(\|O\|=(?P<ev_O>\d+) f=[0-9\.]+ dT=\d+\)", line)
                if tmp is not None:
                    more_stats["evs_O"].append(int(tmp.group("ev_O")))
                else:
                    tmp = re.match(
                        "\-\- Data Sequence \|A\|=(?P<size_ABC>\d+) \|O\|=(?P<size_O>\d+) dT=(?P<size_T>\d+)", line)
                    if tmp is not None:
                        more_stats.update(
                            dict([(k, int(v)) for k, v in tmp.groupdict().items()]))

    times = {}
    selects = {}
    for line in time_lines + inter_lines:
        tmp = re.match(
            "(\[[A-Z]+\] )?(?P<what>\S+) selection \((?P<nb_cands>[0-9]+) candidates\)", line)
        if tmp is not None:
            what = tmp.group("what")
            if what not in selects:
                selects[what] = {}
            selects[what]["nb_cands"] = int(tmp.group("nb_cands"))
        else:
            tmp = re.match(
                "(\[[A-Z]+\] )?(?P<what>\S+) selection done in (?P<time>[0-9:\.]+) at", line)
            if tmp is not None:
                what = tmp.group("what")
                if what not in selects:
                    selects[what] = {}
                selects[what]["time"] = timedeltastr_to_num(tmp.group("time"))

        tmp = re.match(
            "(\[TIME\] )(?P<what>.*) done in (?P<time>[0-9:\.]+)", line)
        if tmp is not None:
            ttm = re.match(
                "(\[TIME\] )Mining done in [0-9:\.]+ \(-inter=(?P<time>[0-9:\.]+)\)", line)
            if ttm is not None:
                times["Mining_only"] = timedeltastr_to_num(tmp.group("time"))
            times[tmp.group("what")] = timedeltastr_to_num(tmp.group("time"))
    try:
        more_stats["nb_rounds"] = max(
            [int(k.split()[-1]) for k in times.keys() if re.match("Combination round \d+$", k)])
    except ValueError:
        print(fn)
        pdb.set_trace()
    X = numpy.array(more_stats.pop("evs_O"))
    more_stats.update(dict([("evNbO_>10", numpy.sum(X > 10, axis=0))] + [("evNbO_%s" % pref, nfun(X)) for pref, nfun in
                                                                         [("median", numpy.median),
                                                                          ("mean", numpy.mean), ("max", numpy.max)]]))
    return times, selects, more_stats


def parse_patts_stats(fn):
    stats_lines = []

    match_patts = ["Code length patterns \((?P<nb_patts>[0-9]*)\): (?P<patt_cl>[0-9\.]*)",
                   "Code length residuals \((?P<nb_residuals>[0-9]*)\): (?P<residuals_cl>[0-9\.]*)",
                   "\-\- Total code length = (?P<compressed_cl>[0-9\.]*) \((?P<prc_cl>[0-9\.]*)% of (?P<original_cl>[0-9\.]*)\)"]
    dt = {}
    patts_dt = []
    with open(fn) as fp:
        for line in fp:
            tmp = re.match("t0=\d+\t(?P<ptree>[^\t]+)\tCode length:.*Occs \((?P<cov_occs_dup>\d+)/(?P<cov_occs>\d+)\)",
                           line)
            if tmp is not None:
                rmax = 0
                for tt in re.finditer("r=(?P<r>\d+)", tmp.group("ptree")):
                    r = int(tt.group("r"))
                    if r > rmax:
                        rmax = r
                patts_dt.append((rmax, int(tmp.group("cov_occs")),
                                int(tmp.group("cov_occs_dup"))))
            else:
                tmp = re.match(" \-\-\-\- COLLECTION STATS \(Total\=(?P<nb_total>[0-9]*) (?P<nbs>[a-z0-9\_\= ]*)\)",
                               line)
                if tmp is not None:
                    dt["total"] = tmp.group("nb_total")
                    dt.update(dict([tt.split("=")[:2]
                              for tt in tmp.group("nbs").split(" ")]))
                else:
                    for match_patt in match_patts:
                        tmp = re.match(match_patt, line)
                        if tmp is not None:
                            dt.update(tmp.groupdict())
    dt = dict([(k, float(v)) for (k, v) in dt.items()])
    X = numpy.array(patts_dt)
    dt.update(dict(zip(*[["%s_%s" % (c, ">3")
              for c in ["r", "nbO", "nbOdup"]], numpy.sum(X > 3, axis=0)])))
    for pref, nfun in [("median", numpy.median), ("mean", numpy.mean), ("max", numpy.max)]:
        dt.update(dict(zip(*[["%s_%s" % (c, pref)
                  for c in ["r", "nbO", "nbOdup"]], nfun(X, axis=0)])))
    return dt


def format_stats_head(add_stats=[], mks=[]):
    entries = ["basis"]
    for (short, imain, isec) in INTERMS:
        entries.extend(["%s_%s" % (short, main_stat)
                       for main_stat in MAIN_STATS])
        entries.extend(["%s_%s" % (short, add_stat) for add_stat in add_stats])
        entries.extend(["%s_%s" % (short, sec_stat) for sec_stat in SEC_STATS])
    entries.extend(mks)
    entries.extend(["runtime_%s" % short for (short, main_time) in MAIN_TIMES])
    return " ".join(entries) + "\n"


def format_stats_row(basis, times, selects, more_stats, inter_dt, add_stats=[]):
    entries = []
    for (short, imain, isec) in INTERMS:
        entries.extend([inter_dt[imain].get(main_stat, 0)
                       for main_stat in MAIN_STATS])
        entries.extend([inter_dt[imain].get(add_stat, -1)
                       for add_stat in add_stats])
        entries.extend([selects[isec][sec_stat] for sec_stat in SEC_STATS])
    entries.extend([more_stats[mk] for mk in sorted(more_stats.keys())])
    entries.extend([times[main_time] for (short, main_time) in MAIN_TIMES])
    return " ".join([basis] + ["%s" % e for e in entries]) + "\n"


# fo = sys.stdout
fo = open(XPS_REP + OUT_FILE, "w")
add_stats = None

for fn in sorted(glob.glob(XPS_REP + XPS_SUB + LOG_MATCH)):
    tmp = re.search("/(?P<basis>[^/]*)_log.txt", fn)
    if tmp is not None:
        basis = tmp.group("basis")
        times, selects, more_stats = parse_log_times(fn)
        inter_dt = {}

        for ppfn in glob.glob(XPS_REP + XPS_SUB + basis + PATT_MATCH):
            tmpp = re.search("/" + basis + "_(?P<inter>patts.*).txt", ppfn)
            if tmpp is not None:
                inter_dt[tmpp.group("inter")] = parse_patts_stats(ppfn)

        if add_stats is None:
            add_stats = sorted(set(list(inter_dt.values())[
                               0].keys()).difference(MAIN_STATS))
            mks = sorted(more_stats.keys())
            fo.write(format_stats_head(add_stats, mks))

        fo.write(format_stats_row(basis, times, selects,
                 more_stats, inter_dt, add_stats))
fo.close()
