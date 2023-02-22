import glob
import os
import re
import sys

import numpy

from read_data import readSequence
from skmine.periodic.data_sequence import DataSequence
from skmine.periodic.pattern import Pattern
from skmine.periodic.pattern_collection import PatternCollection
from skmine.periodic.run_mine import mine_seqs

CMP_OUT_CODES = ["Perfect match",
                 "Patts cover match, same code length",
                 "Patts cover match, better code length",
                 "Patts cover match, some worse, worse code length",
                 "Patts cover match, some worse, better code length",
                 "Patts cover match, some worse, same code length",
                 "Overall cover match, worse code length",
                 "Overall cover match, better code length",
                 "Overall cover match, same code length",
                 "No cover match, worse code length",
                 "No cover match, better code length",
                 "No cover match, same code length",
                 "No pattern found, worse code length",
                 "No pattern found, better code length",
                 "No pattern found, same code length",
                 "?"]


class SyntheXPS:
    def __init__(self):
        self.collected_pcs = {}

    def addPC(self, pc, ds, fn_basis, suff, fo_log=None):
        self.collected_pcs[suff] = pc

    def getPC(self, suff=""):
        return self.collected_pcs[suff]


def nest_inner(inner, RandPs):
    if len(RandPs) == 0:
        return Pattern.parseTreeStr(inner)
    else:
        patt = nest_inner(inner, RandPs[:-1])
        if RandPs[-1][0] > 1:
            patt.repeat(RandPs[-1][0], RandPs[-1][1])
        return patt
    return None


def prepare_pattern(inner, Rs, Ps, t0=0, noise_lvl=0, noise_dens=0):
    RandPs = list(zip(*[Rs, Ps]))
    Ptree = nest_inner(inner, RandPs)
    occsTEK = Ptree.getOccsStar(time=t0)
    if noise_lvl > 0 and noise_dens > 0:
        E = numpy.array(numpy.random.randint(1, noise_lvl + 1, len(occsTEK) - 1)
                        * numpy.sign(0.5 - numpy.random.random(len(occsTEK) - 1)), dtype=int)
        if noise_dens < 1:
            dd = numpy.random.random(len(occsTEK) - 1) > noise_dens
            E[dd] = 0
        Ed = Ptree.getEDict(occsTEK, E)
        occs = [(o[0] + Ptree.getCCorr(o[-1], Ed), o[1]) for o in occsTEK]
    else:
        E = numpy.zeros(len(occsTEK) - 1, dtype=int)
        occs = [(occsTEK[0][0], occsTEK[0][1])] + [(o[0] + E[i], o[1])
                                                   for i, o in enumerate(occsTEK[1:])]
    return (Ptree, t0, E), occs


def prepare_bck(T_first, T_last, P_occs, event, nb_occs, t_last=1, t_first=0):
    if t_last <= 1:
        tz = int(T_last * t_last)
    else:
        tz = int(t_last)
    if 0 < t_first < 1:
        ta = int(T_first * t_first)
    else:
        ta = int(t_first)
    if nb_occs < 1:
        nb_occs = int(nb_occs * P_occs)
    return [(t, event) for t in numpy.random.choice(numpy.arange(ta, tz), nb_occs, replace=False)]


def prepare_synthetic_seq(patts_dets, bck_dets):
    Hpatts = []
    Hoccs = set()
    for patt_dets in patts_dets:
        patt, occs = prepare_pattern(**patt_dets)
        Hoccs.update(occs)
        Hpatts.append(patt)
    tmp = sorted(Hoccs)
    T_first, T_last = (tmp[0][0], tmp[-1][0])

    for bck_det in bck_dets:
        occs = prepare_bck(T_first, T_last, len(tmp), **bck_det)
        Hoccs.update(occs)

    ds = DataSequence(sorted(Hoccs))
    ev_to_num = ds.getEvToNum()
    for (Ptree, tp0, Ep) in Hpatts:
        Ptree.mapEvents(ev_to_num)
    pc_org = PatternCollection(Hpatts)
    return pc_org, ds


def write_ds(ds, fn):
    with open(fn, "w") as fo:
        fo.write(ds.getSequenceStr())


def write_pc(ds, pc, fn):
    with open(fn, "w") as fo:
        fo.write(pc.strPatternsTriples(ds))


def load_pc(fn, ds=None):
    patts = []
    with open(fn) as fp:
        for line in fp:
            if not re.match("#", line):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    patts.append((Pattern.parseTreeStr(parts[0], leaves_first=True), int(
                        parts[1]), list(map(int, parts[2].split()))))

    if ds is not None:
        ev_to_num = ds.getEvToNum()
        for (Ptree, tp0, Ep) in patts:
            Ptree.mapEvents(ev_to_num)
    return PatternCollection(patts)


def match_score_sets(A, B):
    return jaccard(A, B)


def jaccard(A, B):
    return float(len(A.intersection(B))) / len(A.union(B))


def compare_pcs(ds, Hpc, Fpc):
    data_details = ds.getDetails()
    stats = {0: {}, 1: {}}
    for (pi, pc) in [(0, Hpc), (1, Fpc)]:
        stats[pi]["PC"] = pc
        stats[pi]["nbP"] = len(pc.getPatterns())
        stats[pi]["Pcl"] = [p.codeLength(t0, E, data_details)
                            for (p, t0, E) in pc.getPatterns()]
        stats[pi]["trees"] = ["%s" % p[0] for p in pc.getPatterns()]
        stats[pi]["occs"] = [set(o) for o in pc.getOccLists()]
        stats[pi]["cov"] = pc.getCoveredOccs()
        stats[pi]["cl"] = pc.codeLength(ds)
    stats["cl_diff"] = stats[0]["cl"] - stats[1]["cl"]
    stats["sc_cov"] = match_score_sets(stats[0]["cov"], stats[1]["cov"])

    incl_HinF = numpy.array([[stats[0]["occs"][i].issubset(stats[1]["occs"][j])
                              for j in range(stats[1]["nbP"])] for i in range(stats[0]["nbP"])])
    incl_FinH = numpy.array([[stats[1]["occs"][j].issubset(stats[0]["occs"][i])
                              for j in range(stats[1]["nbP"])] for i in range(stats[0]["nbP"])])

    iden = {"ids": []}
    if stats[1]["nbP"] > 0:
        iden["ids"] = list(zip(*numpy.where(incl_HinF & incl_FinH)))
        # "HinF": {"ids" : list(zip(*numpy.where(incl_HinF & ~incl_FinH)))},
        # "FinH": {"ids" : list(zip(*numpy.where(~incl_HinF & incl_FinH)))},
        # "other": {"ids" : list(zip(*numpy.where(~incl_HinF & ~incl_FinH)))}

    ax = 0
    if len(iden["ids"]) > 0:
        iden["cl_diff"] = numpy.array(
            [stats[ax]["Pcl"][ai] - stats[1 - ax]["Pcl"][bi] for (ai, bi) in iden["ids"]])
        iden["tree_cmp"] = numpy.array(
            [stats[ax]["trees"][ai] == stats[1 - ax]["trees"][bi] for (ai, bi) in iden["ids"]])
        iden["occs_cmp"] = numpy.array([match_score_sets(
            stats[ax]["occs"][ai], stats[1 - ax]["occs"][bi]) for (ai, bi) in iden["ids"]])

    out_v = -1
    if stats[1]["nbP"] > 0 and len(iden["ids"]) > 0:
        if numpy.all(iden["cl_diff"] == 0):
            if numpy.all(iden["tree_cmp"]):
                out_v = 0
            else:
                out_v = 1
        elif numpy.all(iden["cl_diff"] > 0):
            out_v = 2
        else:
            out_v = 3 + 1 * (stats["cl_diff"] == 0) + 1 * (stats["cl_diff"] >= 0)
    else:
        if stats[1]["nbP"] == 0:
            offset = 12
        elif stats["sc_cov"] == 1:
            offset = 6
        else:
            offset = 9
        out_v = offset + 1 * (stats["cl_diff"] == 0) + 1 * (stats["cl_diff"] >= 0)
    return out_v, stats, iden


def writeSYNTHin(ds, pcH, fn_basis):
    if fn_basis != "-":
        write_ds(ds, fn_basis + "_ds.txt")
        write_pc(ds, pcH, fn_basis + "_pcH.txt")


def writeSYNTHout(setts, ds, pcH, pcF, out_v, fn_basis, save_pc=True, comb_setts=None):
    if fn_basis is None:
        return
    if fn_basis == "-":
        fo_patts = sys.stdout
    else:
        fo_patts = open(fn_basis + "_summary.txt", "w")

    fo_patts.write("%d >> %s\n" % (out_v, CMP_OUT_CODES[out_v]))
    fo_patts.write("=== SETTINGS ===\n%s\n" % setts)
    if comb_setts is not None:
        fo_patts.write("%s\n" % comb_setts)
    fo_patts.write("=== HIDDEN ===\n%s%s" % pcH.strDetailed(ds))
    fo_patts.write("=== FOUND ===\n%s%s" % pcF.strDetailed(ds))
    if fn_basis != "-":
        if write_pc:
            write_pc(ds, pcF, fn_basis + "_pcF.txt")
        fo_patts.close()


def run_one(setts, fn_b, i, counts_cmp):
    fn_basis = "%s-%s" % (fn_b, i)
    if os.path.isfile(fn_basis + "_pcH.txt"):
        ds = DataSequence(readSequence(
            {"filename": fn_basis + "_ds.txt", "SEP": " "}))
        pcH = load_pc(fn_basis + "_pcH.txt", ds)
    else:
        # generate data sequence
        k = setts["level"]
        Rs = []
        if k == 1:
            Rs = [numpy.random.randint(
                int(.66 * setts["nb_occs"]), setts["nb_occs"])]
        elif k > 1:
            mm = int(numpy.floor(
                (setts["nb_occs"] / (1. * numpy.prod(range(1, k + 1)))) ** (1. / k)))
            if mm < 3:
                nn = int(numpy.floor(setts["nb_occs"] ** (1. / k)))
                if nn < 3:
                    Rs = [3 for kk in range(k)]
                else:
                    xx = [(.33 * nn, nn + 1) for kk in range(k)]
                    Rs = [numpy.random.randint(
                        max(3, numpy.ceil(.33 * nn)), nn + 1) for kk in range(k)][::-1]
            else:
                xx = [((kk + .33) * mm, (kk + 1.) * mm) for kk in range(k)]
                Rs = [numpy.random.randint(
                    max(3, numpy.ceil((kk + .33) * mm)), (kk + 1) * mm + 1) for kk in range(k)][::-1]
        Ps = [numpy.random.randint(setts["p_down"], setts["p_up"])]
        for kk in range(1, k):
            prev = Ps[-1] * .5 * Rs[kk - 1]
            if not setts.get("overlap", False):
                prev = Ps[-1] * Rs[kk - 1]
            tt = numpy.random.randint(prev, prev + 100)
            ik = 0
            while any([tt % p == 0 for p in Ps]) and ik < 100:
                ik += 1
                tt = numpy.random.randint(prev, prev + 100)
            Ps.append(tt)
        patts_dets = [{"inner": setts["inner"], "t0": 0, "Rs": Rs, "Ps": Ps,
                       "noise_lvl": setts["noise_lvl"], "noise_dens": setts["noise_dens"]}]
        bck_dets = []
        for (e, c) in setts.get("add_noise", []):
            bck_dets.append({"event": e, "nb_occs": c})

        print(patts_dets, bck_dets, numpy.prod(Rs))
        pcH, ds = prepare_synthetic_seq(patts_dets, bck_dets)
        writeSYNTHin(ds, pcH, fn_basis)
        #####

    save_pcF = True
    if os.path.isfile(fn_basis + "_pcF.txt"):
        pcF = load_pc(fn_basis + "_pcF.txt", ds)
        save_pcF = False
    else:
        # mine data sequence
        SXPS = SyntheXPS()
        mine_seqs(ds, fn_basis, writePCout_fun=SXPS.addPC)
        pcF = SXPS.getPC()
        #####
    if os.path.isfile(fn_basis + "_summary.txt"):
        out_v = -1
        with open(fn_basis + "_summary.txt") as fp:
            out_v = int(fp.readline().split()[0])
        stats = {0: {"cl": pcH.codeLength(ds)}, 1: {"cl": pcF.codeLength(ds)}}
    else:
        out_v, stats, results = compare_pcs(ds, pcH, pcF)
        writeSYNTHout(setts, ds, pcH, pcF, out_v, fn_basis, save_pc=save_pcF)
    counts_cmp[out_v] = counts_cmp.get(out_v, 0) + 1

    if out_v != 0:  # == -1:
        print("RUN %s\tcl: %f vs. %f\t%d >> %s" %
              (i, stats[0]["cl"], stats[1]["cl"], out_v, CMP_OUT_CODES[out_v]))
    return stats[0]["cl"], stats[1]["cl"], ds.codeLengthResiduals()


def run_combine(setts, fn_b, i, counts_cmp, pool):
    fn_basis = "%s-%s" % (fn_b, i)

    k = numpy.random.randint(setts["k_low"], setts["k_up"] + 1)
    patt_fn = []
    combine_seqs = {}
    patterns_list = []
    offset_t0 = 0
    prev_span = 0
    t0_list = []
    for ii in numpy.random.choice(len(pool), size=k):
        fn_sub_basis = re.sub("_ds.txt", "", pool[ii])
        seqs = readSequence({"filename": fn_sub_basis + "_ds.txt", "SEP": " "})
        ds = DataSequence(seqs)
        pcsH = load_pc(fn_sub_basis + "_pcH.txt", ds)
        patt_fn.append(fn_sub_basis)

        next_t0 = offset_t0 + numpy.random.randint(numpy.ceil(
            prev_span * setts["t0_low"]), numpy.ceil(prev_span * setts["t0_up"]) + 1)
        t0_list.append(next_t0)
        for ev, seq in seqs.items():
            if ev not in combine_seqs:
                combine_seqs[ev] = set()
            combine_seqs[ev].update(seq + next_t0)

        for (p, pt0, pE) in pcsH.getPatterns():
            patterns_list.append((p, pt0 + next_t0, pE))

        offset_t0 = next_t0
        prev_span = ds.getTend()

    comb_setts = {"t0s": t0_list, "patt_fn": patt_fn, "k": k}
    combine_ss = dict([(ev, numpy.array(sorted(s)))
                       for (ev, s) in combine_seqs.items()])
    ds = DataSequence(combine_ss)
    pcH = PatternCollection(patterns_list)
    writeSYNTHin(ds, pcH, fn_basis)

    # mine data sequence
    SXPS = SyntheXPS()
    mine_seqs(ds, fn_basis, writePCout_fun=SXPS.addPC)
    pcF = SXPS.getPC()
    #####
    out_v, stats, results = compare_pcs(ds, pcH, pcF)
    writeSYNTHout(setts, ds, pcH, pcF, out_v, fn_basis,
                  save_pc=True, comb_setts=comb_setts)
    counts_cmp[out_v] = counts_cmp.get(out_v, 0) + 1

    if out_v != 0:  # == -1:
        print("RUN %s\tcl: %f vs. %f\t%d >> %s" %
              (i, stats[0]["cl"], stats[1]["cl"], out_v, CMP_OUT_CODES[out_v]))
    return stats[0]["cl"], stats[1]["cl"], ds.codeLengthResiduals()


def run_simple(series_basis, xps_rep, nb_runs):
    # run_id = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    series_patts = [{"inner": "a", "max_level": 3, "p_down": 10,
                     "p_up": 25, "max_noise_lvl": 3, "max_noise_dens": 3, "nb_occs": 250},
                    {"inner": "a", "max_level": 3, "p_down": 5,
                     "p_up": 10, "max_noise_lvl": 2, "max_noise_dens": 3, "nb_occs": 500},
                    {"inner": "a [d=4] b", "max_level": 3, "p_down": 10,
                     "p_up": 25, "max_noise_lvl": 1, "max_noise_dens": 1, "nb_occs": 500},
                    {"inner": "a [d=1] c [d=2] d", "max_level": 3, "p_down": 10,
                     "p_up": 25, "max_noise_lvl": 1, "max_noise_dens": 1, "nb_occs": 500}]  # , "add_noise": [("z", .1)

    ss = []
    for j, serie_patts in enumerate(series_patts):
        for level in range(serie_patts["max_level"]):
            for noise_lvl in range(serie_patts["max_noise_lvl"] + 1):
                nd = [0.]
                if noise_lvl > 0:
                    nd = range(1, serie_patts["max_noise_dens"] + 1)
                for noise_dens in nd:
                    ss.append({"j": j, "noise_lvl": noise_lvl, "noise_dens": noise_dens / 10.,
                               "inner": serie_patts["inner"], "level": level + 1, "nb_occs": serie_patts["nb_occs"],
                               "p_down": serie_patts["p_down"], "p_up": serie_patts["p_up"],
                               "overlap": serie_patts.get("overlap", False)})
                    if "add_noise" in serie_patts:
                        ss[-1]["add_noise"] = serie_patts["add_noise"]

                    if re.search("V", series_basis):
                        ss[-1]["add_noise"] = [("a", .1)]
                    elif re.search("W", series_basis):
                        ss[-1]["add_noise"] = [("a", .5)]
                    elif re.search("U", series_basis):
                        ss[-1]["overlap"] = True

    for si, setts in enumerate(ss):
        print(setts)
        fn_b = "%s%s%s-%s" % (xps_rep, series_basis, setts["j"], si)
        counts_cmp = {}
        cl_pairs = []
        for i in range(nb_runs):
            cl_pair = run_one(setts, fn_b, i, counts_cmp)
            cl_pairs.append(cl_pair)

        with open("%s_series-summary.txt" % fn_b, "w") as fo:
            fo.write("=== SETTINGS ===\n%s\n" % setts)
            for (v, c) in sorted(counts_cmp.items()):
                print("%d/%d\t(%.3f)\t%d << %s" %
                      (c, nb_runs, c / float(nb_runs), v, CMP_OUT_CODES[v]))
                fo.write("%d/%d\t(%.3f)\t%d << %s\n" %
                         (c, nb_runs, c / float(nb_runs), v, CMP_OUT_CODES[v]))
            numpy.savetxt(fo, numpy.array(cl_pairs), fmt='%f')


def run_comb(series_basis, xps_rep, nb_runs):
    ss = [{"k_low": 2, "k_up": 5, "t0_low": 1.2, "t0_up": 1.5},
          {"k_low": 2, "k_up": 5, "t0_low": .2, "t0_up": .9}]
    match_pool = "%s%s[0-9\-]*_ds.txt" % (xps_rep, series_basis)
    pool = glob.glob(match_pool)

    for si, setts in enumerate(ss):
        print(setts)
        fn_b = "%s%s%s-%s" % (xps_rep, series_basis, "comb", si)
        counts_cmp = {}
        cl_pairs = []
        for i in range(nb_runs):
            cl_pair = run_combine(setts, fn_b, i, counts_cmp, pool)
            cl_pairs.append(cl_pair)

        with open("%s_series-summary.txt" % fn_b, "w") as fo:
            fo.write("=== SETTINGS ===\n%s\n" % setts)
            for (v, c) in sorted(counts_cmp.items()):
                print("%d/%d\t(%.3f)\t%d << %s" %
                      (c, nb_runs, c / float(nb_runs), v, CMP_OUT_CODES[v]))
                fo.write("%d/%d\t(%.3f)\t%d << %s\n" %
                         (c, nb_runs, c / float(nb_runs), v, CMP_OUT_CODES[v]))
            numpy.savetxt(fo, numpy.array(cl_pairs), fmt='%f')


if __name__ == "__main__":
    BASIS_REP = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    XPS_REP = BASIS_REP + "/xps/synthe/"

    # run_id = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    NB_RUNS = 20
    for series_basis in ["synthe_S", "synthe_V", "synthe_W", "synthe_U"]:
        run_simple(series_basis, XPS_REP, NB_RUNS)

    # needs the previous to generate basis event sequences to merge
    NB_RUNS = 100
    for series_basis in ["synthe_S", "synthe_V", "synthe_W", "synthe_U"]:
        run_comb(series_basis, XPS_REP, NB_RUNS)
