import glob
import os
import re

import matplotlib.pyplot as plt
import numpy

FIELDS = ["level", "noise_lvl", "noise_dens"]
FIELDS_ORD = ["cl_res", "cl_H", "prc_cl_H", "cl_F", "prc_cl_F"] + \
             ["inner_p"] + FIELDS + ["nb_miss_p", "max_miss_pv"]
FIELDS_MAP = dict([(v, k) for (k, v) in enumerate(FIELDS_ORD)])

FIELDS_COMB = ['t0_low', 't0_up', 'k_low', 'k_up', "seriesL", "seriesX"]
FIELDS_ORD_COMB = ["cl_res", "cl_H", "prc_cl_H", "cl_F",
                   "prc_cl_F"] + FIELDS_COMB + ["nb_miss_p", "max_miss_pv"]
FIELDS_MAP_COMB = dict([(v, k) for (k, v) in enumerate(FIELDS_ORD_COMB)])

colors_all = {0.: "#332288", .1: "#88CCEE", .2: "#44AA99",
              0.3: "#DDCC77"}  # , .4:"#CC6677"}
colors_comb = {0.2: "#332288", 1.2: "#CC6677"}


def get_params_vs(collected_results, series):
    if "inner" not in collected_results[series]["params"]:
        return [collected_results[series]["params"].get(kk, -1) for kk in FIELDS_COMB]
    elif collected_results[series]["params"]["inner"] == "a":
        if collected_results[series]["params"]["p_down"] == 10:
            inner_p = 0
        else:
            inner_p = 1
    elif "b" in collected_results[series]["params"]["inner"]:
        inner_p = 2
    else:
        inner_p = 3
    return [inner_p] + [collected_results[series]["params"].get(kk, -1) for kk in FIELDS]


def collect_results(xps_rep, series_basis, mtch):
    collected_results = {}
    match_summaries = "%s%s%s[A-Za-z0-9\-]*_summary.txt" % (
        xps_rep, series_basis, mtch)
    for fi, fn in enumerate(glob.glob(match_summaries)):
        tmp = re.search(
            "_(?P<seriesL>[UVWS])(comb)?(?P<series>[0-9\-]+)_summary\.txt", fn)
        if tmp is not None:
            which = "patts_H"
            series = tmp.group("series")
            seriesX = int(series.split("-")[1])
            seriesL = ord(tmp.group("seriesL"))
            collected_results[series] = {"cls": [], "patts_H": [
            ], "patts_F": [], "ps_H": set(), "ps_F": set()}
            with open(fn) as fp:
                for line in fp:
                    ttmp = re.search(
                        "Total code length = (?P<cl>[0-9\.]+) \((?P<prc_cl>[0-9\.]+)\% of (?P<cl_res>[0-9\.]+)\)", line)
                    if ttmp is not None:
                        if len(collected_results[series]["cls"]) == 0:
                            collected_results[series]["cls"] = list(
                                map(float, [ttmp.group("cl_res"), ttmp.group("cl"), ttmp.group("prc_cl")]))
                            which = "H"
                        else:
                            collected_results[series]["cls"].extend(
                                list(map(float, [ttmp.group("cl"), ttmp.group("prc_cl")])))
                            which = "F"
                    else:
                        ttmp = re.match(
                            "t0=[0-9]+\t(?P<patt>.+)\tCode length", line)
                        if ttmp is not None:
                            pt = ttmp.group("patt")
                            collected_results[series]["patts_%s" %
                                                      which].append(pt)
                            for t in re.finditer("p=(?P<p>[0-9]+)\]", pt):
                                collected_results[series]["ps_%s" % which].add(
                                    int(t.group("p")))
                        elif re.match("{", line):
                            if "params" not in collected_results[series]:
                                collected_results[series]["params"] = eval(
                                    line.strip())
                                collected_results[series]["params"]["seriesL"] = seriesL
                                collected_results[series]["params"]["seriesX"] = seriesX
                            else:
                                collected_results[series]["params"].update(
                                    eval(line.strip()))
    return collected_results


def make_plots(collected_results, series_basis):
    Blbls = {"a": "a", "a": "a", 'a [d=4] b': "a $-$4$-$ b",
             'a [d=1] c [d=2] d': "a $-$1$-$ c $-$2$-$ d"}
    blocks = [("a", 5), ("a", 10), ('a [d=4] b', 10),
              ('a [d=1] c [d=2] d', 10)]

    ks = sorted(collected_results.keys())
    mat_dt = numpy.array([collected_results[series]["cls"] +
                          get_params_vs(collected_results, series) + [0, 0] for series in ks])
    for i, k in enumerate(ks):
        ps_F, ps_H = (collected_results[k]["ps_F"],
                      collected_results[k]["ps_H"])
        ps_diff = ps_F.difference(ps_H)
        mat_dt[i, FIELDS_MAP["nb_miss_p"]] = len(ps_diff)
        collected_results[k]["ps_diff"] = ps_diff
        if len(ps_diff) > 0:
            tt = [numpy.min(numpy.abs([pH - p for pH in ps_H])) for p in ps_diff]
            mat_dt[i, FIELDS_MAP["max_miss_pv"]] = numpy.max(tt)

    levels = [1, 2, 3]
    inners = range(len(blocks))
    ni = 0
    # ,sharex="col", sharey="row")
    fig, grid = plt.subplots(len(levels), len(inners), figsize=(11, 7))
    (xmin, xmax) = numpy.min(mat_dt[:, FIELDS_MAP["prc_cl_H"]]), numpy.max(
        mat_dt[:, FIELDS_MAP["prc_cl_H"]])
    (ymin, ymax) = numpy.min(mat_dt[:, FIELDS_MAP["prc_cl_F"]]), numpy.max(
        mat_dt[:, FIELDS_MAP["prc_cl_F"]])
    for li, level in enumerate(levels):
        for ii, inner in enumerate(inners):
            ni += 1
            rids = ((mat_dt[:, FIELDS_MAP["level"]] == level) &
                    (mat_dt[:, FIELDS_MAP["inner_p"]] == inner))
            (xmin, xmax) = numpy.min(
                mat_dt[(mat_dt[:, FIELDS_MAP["inner_p"]] == inner), FIELDS_MAP["prc_cl_H"]]), numpy.max(
                mat_dt[(mat_dt[:, FIELDS_MAP["inner_p"]] == inner), FIELDS_MAP["prc_cl_H"]])

            grid[li, ii].plot([0, 100], [0, 100], linestyle=(
                0, (1, 1)), linewidth=.5, color="darkgray")
            grid[li, ii].plot([0, 100], [0, 150], linestyle=(
                0, (2, 2)), linewidth=.5, color="darkgray")
            grid[li, ii].plot([0, 100], [0, 50], linestyle=(
                0, (2, 2)), linewidth=.5, color="darkgray")
            grid[li, ii].plot([0, 100], [0, 200], linestyle=(
                0, (4, 4)), linewidth=.5, color="darkgray")

            for nd in numpy.unique(mat_dt[rids, FIELDS_MAP["noise_dens"]]):
                rrs = rids & (mat_dt[:, FIELDS_MAP["noise_dens"]] == nd)
                for nl in numpy.unique(mat_dt[rrs, FIELDS_MAP["noise_lvl"]]):
                    iids = rrs & (mat_dt[:, FIELDS_MAP["noise_lvl"]] == nl)
                    grid[li, ii].plot(mat_dt[iids, FIELDS_MAP["prc_cl_H"]], mat_dt[iids, FIELDS_MAP["prc_cl_F"]],
                                      "o", mec=colors_all[nd], color=colors_all[nd], markersize=3 + 1 * nl, alpha=0.8,
                                      zorder=10)

            grid[li, ii].set_xlim([numpy.floor(xmin) - .9, numpy.ceil(xmax) + .9])
            grid[li, ii].set_ylim([numpy.floor(ymin) - .9, numpy.ceil(ymax) + .9])
            tt = grid[li, ii].get_xticks()
            if len(tt) > 6:
                grid[li, ii].set_xticks(tt[1:-1:2])
            tt = grid[li, ii].get_yticks()
            grid[li, ii].set_yticks(tt[1:-1])
            if li == 0:
                grid[li, ii].set_title("{p>%s} (%s)" % (
                    blocks[inner][1], Blbls[blocks[inner][0]]), fontsize=14)
            if li < 2:
                grid[li, ii].set_xticklabels([])
            if ii > 0:
                grid[li, ii].set_yticklabels([])
            if ii == 3:
                grid[li, ii].yaxis.set_label_position("right")
                grid[li, ii].set_ylabel("height=%d" % level, fontsize=14)

    fig.text(0.03, 0.55, "$\%\mathit{L}_F$",
             fontsize=14, va='center', rotation='vertical')
    fig.text(0.5, 0.12, "$\%\mathit{L}_H$", fontsize=14, ha='center')

    plt.subplots_adjust(wspace=.03, hspace=.03, bottom=0.2,
                        top=0.92, right=0.92, left=0.08)
    kks = sorted(colors_all.keys())
    lls = [plt.plot(-1, -1, color=colors_all[kk], linewidth=10)[0]
           for kk in kks]
    labels = ["%.1f" % kk for kk in kks]
    labels[0] = "shift noise density:     " + labels[0]

    bb = (fig.subplotpars.left, fig.subplotpars.bottom - 0.18,
          fig.subplotpars.right - fig.subplotpars.left, .1)
    grid[0, 0].legend(lls, labels, bbox_to_anchor=bb, mode="expand", loc="lower left", frameon=False, markerfirst=False,
                      borderaxespad=0., ncol=4, bbox_transform=fig.transFigure)

    plt.savefig("%s%s_scatter.pdf" % (PLT_REP, series_basis))


def make_combplots(collected_results, series_basis):
    lbls_series = {"S": "No additive noise", "U": "Interleaving",
                   "V": "Additive noise (a, .1)", "W": "Additive noise (a, .5)"}

    ks = sorted(collected_results.keys())
    mat_dt = numpy.array([collected_results[series]["cls"] +
                          get_params_vs(collected_results, series) + [0, 0] for series in ks])
    for i, k in enumerate(ks):
        ps_F, ps_H = (collected_results[k]["ps_F"],
                      collected_results[k]["ps_H"])
        ps_diff = ps_F.difference(ps_H)
        mat_dt[i, FIELDS_MAP_COMB["nb_miss_p"]] = len(ps_diff)
        collected_results[k]["ps_diff"] = ps_diff
        if len(ps_diff) > 0:
            tt = [numpy.min(numpy.abs([pH - p for pH in ps_H])) for p in ps_diff]
            mat_dt[i, FIELDS_MAP_COMB["max_miss_pv"]] = numpy.max(tt)

    seriesLs = [ord(cc) for cc in ["S", "V", "W", "U"]]
    seriesXs = ["x"]
    ni = 0
    fig, grid = plt.subplots(len(seriesXs), len(seriesLs), figsize=(14, 4))
    (xmin, xmax) = numpy.min(mat_dt[:, FIELDS_MAP_COMB["prc_cl_H"]]), numpy.max(
        mat_dt[:, FIELDS_MAP_COMB["prc_cl_H"]])
    (ymin, ymax) = numpy.min(mat_dt[:, FIELDS_MAP_COMB["prc_cl_F"]]), numpy.max(
        mat_dt[:, FIELDS_MAP_COMB["prc_cl_F"]])
    for li, seriesL in enumerate(seriesLs):
        for ii, xx in enumerate(seriesXs):
            ni += 1
            rids = (mat_dt[:, FIELDS_MAP_COMB["seriesL"]] == seriesL)
            (xmin, xmax) = numpy.min(mat_dt[:, FIELDS_MAP_COMB["prc_cl_H"]]), numpy.max(
                mat_dt[:, FIELDS_MAP_COMB["prc_cl_H"]])

            grid[li].plot([0, 100], [0, 100], linestyle=(
                0, (1, 1)), linewidth=.5, color="darkgray")
            grid[li].plot([0, 100], [0, 150], linestyle=(
                0, (2, 2)), linewidth=.5, color="darkgray")
            grid[li].plot([0, 100], [0, 50], linestyle=(
                0, (2, 2)), linewidth=.5, color="darkgray")
            grid[li].plot([0, 100], [0, 200], linestyle=(
                0, (4, 4)), linewidth=.5, color="darkgray")

            for nd in numpy.unique(mat_dt[rids, FIELDS_MAP_COMB["t0_low"]]):
                iids = rids & (mat_dt[:, FIELDS_MAP_COMB["t0_low"]] == nd)
                grid[li].plot(mat_dt[iids, FIELDS_MAP_COMB["prc_cl_H"]], mat_dt[iids, FIELDS_MAP_COMB["prc_cl_F"]],
                              "o", mec=colors_comb[nd], color=colors_comb[nd], markersize=4, alpha=0.8, zorder=10)

            grid[li].set_xlim([numpy.floor(xmin) - .9, numpy.ceil(xmax) + .9])
            grid[li].set_ylim([numpy.floor(ymin) - .9, numpy.ceil(ymax) + .9])
            tt = grid[li].get_xticks()
            if len(tt) > 6:
                grid[li].set_xticks(tt[1:-1:2])
            tt = grid[li].get_yticks()
            grid[li].set_yticks(tt[1:-1])
            if li < -1:
                grid[li].set_xticklabels([])
            else:
                grid[li].set_xlabel("$\%\mathit{L}_H$", fontsize=16)
            if li > 0:
                grid[li].set_yticklabels([])
            else:
                grid[li].set_ylabel("$\%\mathit{L}_F$", fontsize=16)
            grid[li].set_title(lbls_series[chr(int(seriesL))])

    plt.subplots_adjust(wspace=.03, hspace=.03,
                        bottom=0.25, top=0.9, right=0.95)
    kks = sorted(colors_comb.keys(), reverse=True)
    lls = [plt.plot(-1, -1, color=colors_comb[kk], linewidth=10)[0]
           for kk in kks]
    labels = ["no overlap", "overlap"]

    bb = (fig.subplotpars.left, fig.subplotpars.bottom - 0.23,
          fig.subplotpars.right - fig.subplotpars.left, .1)
    grid[0].legend(lls, labels, bbox_to_anchor=bb, mode="expand", loc="lower left", frameon=False, markerfirst=False,
                   borderaxespad=0., ncol=2, bbox_transform=fig.transFigure)
    plt.savefig("%s%s_scatter.pdf" % (PLT_REP, series_basis))


if __name__ == "__main__":
    BASIS_REP = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    XPS_REP = BASIS_REP + "/xps/synthe/"
    PLT_REP = BASIS_REP + "/xps/"

    for series_basis in ["synthe_S", "synthe_V", "synthe_W", "synthe_U"]:
        xps_rep = XPS_REP  # + "summaries/"
        collected_results = collect_results(xps_rep, series_basis, "[0-9]")
        make_plots(collected_results, series_basis)

    xps_rep = XPS_REP  # + "summaries_comb/"
    collected_results = collect_results(xps_rep, "synthe_", "*comb")
    make_combplots(collected_results, "synthe_comb")
