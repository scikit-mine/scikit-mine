import numpy
import re
import sys
import datetime
import os
import glob
import matplotlib.pyplot as plt
import pdb

PERFECT_VAL = 0
SCORE_LBL = "$\%\mathit{L}_F - \%\mathit{L}_H$"
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
LBLS_CATS = [chr(x) for x in range(ord("A"), ord("Z"))]
COLORS_CATS_BLUE = ["#3D52A1", "#3A89C9", "#77B7E5", "#B4DDF7", "#E6F5FE"]
COLORS_CATS_RED = ["#FFFAD2", "#FFE3AA",
                   "#F9BD7E", "#ED875E", "#D24D3E", "#AE1C3E"]
COLORS_CATS_GREY = ["#"+6*("%s" % s) for s in range(1, 10)] + \
    ["#"+6*chr(s) for s in range(ord("A"), ord("F"))]
COLORS_CATS_OTHER = ["#7FB972", "#4EB265", "#117755"]

COLORS_CATS = {}
COLORS_CATS.update(dict([(k, COLORS_CATS_GREY[j]) for (j, k) in enumerate(
    [i for i, l in enumerate(CMP_OUT_CODES) if re.search("same", l)])]))
COLORS_CATS.update(dict([(k, COLORS_CATS_RED[j]) for (j, k) in enumerate(
    [i for i, l in enumerate(CMP_OUT_CODES) if re.search("worse", l)])]))
COLORS_CATS.update(dict([(k, COLORS_CATS_BLUE[j]) for (j, k) in enumerate(
    [i for i, l in enumerate(CMP_OUT_CODES) if re.search("better", l)])]))
COLORS_CATS.update(dict([(k, COLORS_CATS_BLUE[j]) for (j, k) in enumerate(
    [i for i, l in enumerate(CMP_OUT_CODES) if re.search("better", l)])]))
COLORS_CATS.update(dict([(k, COLORS_CATS_OTHER[j]) for (j, k) in enumerate(
    sorted(set(range(len(CMP_OUT_CODES))).difference(COLORS_CATS.keys())))]))

COLOR_NLVL = ["#FED98E", "#FB9A29", "#D95F0E", "#993404"]

NLVLS = range(4)
TLVLS = range(1, 4)

all_box_props = {
    "boxprops": {"linestyle": '-', "linewidth": 1, "color": 'black', "facecolor": "yellow"},
    "flierprops": {"marker": '+'},
    "whiskerprops": {"linestyle": '-', "color": "black"},
    "medianprops": {"linestyle": '-', "linewidth": 1.5, "color": 'black'}}


FIELDS = ["level", "noise_lvl", "noise_dens"]
# , 't0_up', 'k_low', 'k_up', "seriesL", "seriesX"]
FIELDS_COMB = ['t0_low', "seriesL"]


def get_params_k(params):
    if "inner" not in params:
        return tuple([params.get(kk, -1) for kk in FIELDS_COMB])
    elif params["inner"] == "a":
        if params["p_down"] == 10:
            inner_p = 0
        else:
            inner_p = 1
    elif "b" in params["inner"]:
        inner_p = 2
    else:
        inner_p = 3
    return tuple([inner_p]+[params.get(kk, -1) for kk in FIELDS])


def rep_collect_results(series_basis, xps_rep):
    collected_results = {}
    match_summaries = "%s%s[0-9\-]*_summary.txt" % (xps_rep, series_basis)
    for fi, fn in enumerate(glob.glob(match_summaries)):
        tmp = re.search(
            "_(?P<seriesL>[UVWS])(comb)?(?P<series_id>[0-9\-]+)_summary\.txt", fn)
        if tmp is not None:
            cl_pairs = []
            patts = {"H": [], "F": []}
            ps = {"H": set(), "F": set()}
            params = {}
            cat = None
            which = "patts_H"
            seriesL = tmp.group("seriesL")
            series_id = tmp.group("series_id")
            with open(fn) as fp:
                for line in fp:
                    ttmp = re.search(
                        "Total code length = (?P<cl>[0-9\.]+) \((?P<prc_cl>[0-9\.]+)\% of (?P<cl_res>[0-9\.]+)\)", line)
                    if ttmp is not None:
                        if len(cl_pairs) == 0:
                            cl_pairs.append(float(ttmp.group("cl")))
                            which = "H"
                        else:
                            cl_pairs.extend(
                                list(map(float, [ttmp.group("cl"), ttmp.group("cl_res")])))
                            which = "F"
                    else:
                        ttmp = re.match(
                            "t0=[0-9]+\t(?P<patt>.+)\tCode length", line)
                        if ttmp is not None:
                            pt = ttmp.group("patt")
                            patts["%s" % which].append(pt)
                            for t in re.finditer("p=(?P<p>[0-9]+)\]", pt):
                                ps["%s" % which].add(int(t.group("p")))
                        elif re.match("{", line):
                            params.update(eval(line.strip()))
                        else:
                            ttt = re.match("(?P<cat>[0-9]+) >>", line)
                            if ttt is not None:
                                cat = int(ttt.group("cat"))
            params["seriesL"] = seriesL
            k = get_params_k(params)
            if k not in collected_results:
                cat_counts = [0 for i in range(len(CMP_OUT_CODES))]
                collected_results[k] = {
                    "params": params, "counts": cat_counts, "cl_pairs": [], "details": []}
            collected_results[k]["counts"][cat] += 1
            collected_results[k]["cl_pairs"].append(cl_pairs)
            collected_results[k]["details"].append(
                {"patts": patts, "ps": ps, "series_id": series_id, "seriesL": seriesL})

    crs = []
    for kk, v in collected_results.items():
        v["cl_pairs"] = numpy.array(v["cl_pairs"])
        crs.append(v)
    return crs


def collect_results(series_basis):
    collected_results = []
    match_summaries = "%s%s[0-9\-]*_series-summary.txt" % (
        XPS_REP, series_basis)
    for fi, fn in enumerate(glob.glob(match_summaries)):
        cl_pairs = []
        tot = -1
        cat_counts = [0 for i in range(len(CMP_OUT_CODES))]
        with open(fn) as fp:
            for li, line in enumerate(fp):
                if li == 1:
                    params = eval(line.strip())
                else:
                    tmp = re.match(
                        "(?P<nb_cat>[0-9]*)/(?P<nb_tot>[0-9]*)\t\([0-9\.]+\)\t(?P<cat_id>[0-9]*) <<", line)
                    if tmp is not None:
                        cat_counts[int(tmp.group("cat_id"))] = int(
                            tmp.group("nb_cat"))

                    elif re.match("^[0-9\. ]*$", line):
                        cl_pairs.append(list(map(float, line.strip().split())))

        params["series"] = "X"
        tmp = re.search("synthe_(?P<series>[SVWU]).*-", fn)
        if tmp is not None:
            params["series"] = tmp.group("series")

        collected_results.append(
            {"params": params, "counts": cat_counts, "cl_pairs": numpy.array(cl_pairs)})
    return collected_results


def draw_synthe_boxes(series_basis):
    xps_rep = XPS_REP  # + "summaries/"
    collected_results = rep_collect_results(series_basis, xps_rep)

    Blbls = {"a": "a", "a": "a", 'a [d=4] b': "a $-$4$-$ b",
             'a [d=1] c [d=2] d': "a $-$1$-$ c $-$2$-$ d"}

    blocks = [("a", 5), ("a", 10), ('a [d=4] b', 10),
              ('a [d=1] c [d=2] d', 10)]
    # vs_all = [(cs["cl_pairs"][:,0]-cs["cl_pairs"][:,1])/cs["cl_pairs"][:,0] for cs in collected_results]
    # vs_all = [numpy.log(cs["cl_pairs"][:,1]/cs["cl_pairs"][:,0]) for cs in collected_results]
    vs_all = [cs["cl_pairs"][:, 1]/cs["cl_pairs"][:, 2] - cs["cl_pairs"]
              [:, 0]/cs["cl_pairs"][:, 2] for cs in collected_results]
    x_min = numpy.floor(10*numpy.min(numpy.hstack(vs_all)))/10.
    x_max = numpy.ceil(10*numpy.max(numpy.hstack(vs_all)))/10.

    nb_c, nb_r = (3, 1)

    fig, grid = plt.subplots(1, 3, figsize=(16, 8))
    for li, level in enumerate(TLVLS):
        offset = 1
        yticks_all = []
        ytick_lbls_all = []
        suboffs = []
        sublbls = []
        for bi, (inner, p_down) in enumerate(blocks):

            rids = [(ci, (p["params"]["noise_lvl"], p["params"]["noise_dens"]), "%s" % (p["params"]["noise_dens"])) for ci, p in enumerate(
                collected_results) if (p["params"]["inner"] == inner and p["params"]["p_down"] == p_down) and (p["params"]["level"] == level)]
            rids.sort(key=lambda x: x[1])

            sdata = [collected_results[r[0]] for r in rids]
            pos = numpy.arange(len(sdata))+offset

            for ni, noise_lvl in enumerate(NLVLS):
                srids = [(ssi, sri[0]) for ssi, sri in enumerate(
                    rids) if sri[1][0] == noise_lvl]
                if len(srids) > 0:
                    vs = [vs_all[r[1]] for r in srids]
                    pps = [pos[r[0]] for r in srids]
                    box = grid[li].boxplot(
                        vs, vert=False, positions=pps, widths=.8, patch_artist=True, **all_box_props)
                    for b in box["boxes"]:
                        b.set_facecolor(COLOR_NLVL[ni])
                        b.set_edgecolor("black")
                    for b in box["whiskers"]:
                        b.set_color(COLOR_NLVL[ni])
                    for b in box["fliers"]:
                        b.set_markeredgecolor(COLOR_NLVL[ni])

            yticks_all.extend(numpy.arange(len(sdata))+offset)
            if li == 0:
                ytick_lbls_all.extend(["%s % 4d/%d" % (r[-1], collected_results[r[0]]["counts"]
                                      [0], numpy.sum(collected_results[r[0]]["counts"])) for r in rids])
            else:
                ytick_lbls_all.extend(["%d/%d" % (collected_results[r[0]]["counts"]
                                      [0], numpy.sum(collected_results[r[0]]["counts"])) for r in rids])

            suboffs.append(offset-1)
            sublbls.append("{p>%s} (%s)" % (p_down, Blbls[inner]))

            offset += len(sdata)+2

        grid[li].set_xlim(x_min, x_max)
        ymin, ymax = (numpy.max(yticks_all)+1, numpy.min(yticks_all)-2)
        grid[li].set_ylim(ymin, ymax)
        grid[li].set_title("height=%d" % level, fontsize=18)
        grid[li].plot([PERFECT_VAL, PERFECT_VAL], [ymin, ymax],
                      ':', color="#999999", zorder=-1)
        grid[li].set_xlabel(SCORE_LBL, fontsize=18)
        if li == 0:
            grid[li].set_ylabel("Noise density", fontsize=14)
        grid[li].set_yticks(yticks_all)
        grid[li].set_yticklabels(ytick_lbls_all, fontsize=14)

        for si in range(len(suboffs)):
            grid[li].text(x_min+(x_max-x_min)/2., suboffs[si],
                          sublbls[si], horizontalalignment='center', fontsize=14)
            grid[li].plot([x_min+.1, x_max-.1], [suboffs[si]+.35,
                          suboffs[si]+.35], "k", linewidth=0.1)

    plt.subplots_adjust(bottom=0.2)
    lls = [plt.plot(-10, -10, COLOR_NLVL[ni], linewidth=10)[0]
           for ni, noise_lvl in enumerate(NLVLS)]
    labels = ["%d" % noise_lvl for ni, noise_lvl in enumerate(NLVLS)]
    labels[0] = "shift noise level:     "+labels[0]

    bb = (fig.subplotpars.left, fig.subplotpars.bottom-0.12,
          fig.subplotpars.right-fig.subplotpars.left, .1)

    grid[0].legend(lls, labels, bbox_to_anchor=bb, mode="expand", loc="lower left", frameon=False, markerfirst=False,
                   borderaxespad=0., ncol=4, bbox_transform=fig.transFigure, fontsize=14)

    plt.draw()
    plt.savefig("%s%s_box.pdf" % (PLT_REP, series_basis))


def draw_synthe_boxes_combs(series_basis):

    lbls_series = {"S": "No additive noise", "U": "Interleaving",
                   "V": "Additive noise (a, .1)", "W": "Additive noise (a, .5)"}

    xps_rep = XPS_REP  # + "summaries_comb/"
    collected_results = rep_collect_results(series_basis, xps_rep)
    blocks = ["S", "V", "W", "U"]
    # vs_all = [(cs["cl_pairs"][:,0]-cs["cl_pairs"][:,1])/cs["cl_pairs"][:,0] for cs in collected_results]
    # vs_all = [numpy.log(cs["cl_pairs"][:,1]/cs["cl_pairs"][:,0]) for cs in collected_results]
    vs_all = [cs["cl_pairs"][:, 1]/cs["cl_pairs"][:, 2] - cs["cl_pairs"]
              [:, 0]/cs["cl_pairs"][:, 2] for cs in collected_results]
    x_min = numpy.floor(10*numpy.min(numpy.hstack(vs_all)))/10.
    x_max = numpy.ceil(10*numpy.max(numpy.hstack(vs_all)))/10.

    lgd = {.2: "overlap", 1.2: "no ov."}

    fig = plt.figure(figsize=(12, 6))
    grid = [plt.subplot(111)]
    for li, sxx in enumerate([0]):
        offset = 1
        yticks_all = []
        ytick_lbls_all = []
        suboffs = []
        sublbls = []
        for bi, sss in enumerate(blocks):

            rids = [(ci, (p["params"]["t0_low"], p["params"]["t0_up"]), lgd.get(p["params"]["t0_low"], "-"))
                    for ci, p in enumerate(collected_results) if (p["params"]["seriesL"] == sss)]
            rids.sort(key=lambda x: x[1])

            sdata = [collected_results[r[0]] for r in rids]
            pos = numpy.arange(len(sdata))+offset

            for ni, noise_lvl in enumerate([(1.2, 1.5), (.2, .9)]):
                srids = [(ssi, sri[0])
                         for ssi, sri in enumerate(rids) if sri[1] == noise_lvl]
                if len(srids) > 0:
                    vs = [vs_all[r[1]] for r in srids]
                    pps = [pos[r[0]] for r in srids]
                    box = grid[li].boxplot(
                        vs, vert=False, positions=pps, widths=.8, patch_artist=True, **all_box_props)
                    for b in box["boxes"]:
                        b.set_facecolor(COLOR_NLVL[ni])
                        b.set_edgecolor("black")
                    for b in box["whiskers"]:
                        b.set_color(COLOR_NLVL[ni])
                    for b in box["fliers"]:
                        b.set_markeredgecolor(COLOR_NLVL[ni])

            yticks_all.extend(numpy.arange(len(sdata))+offset)
            if li == 0:
                ytick_lbls_all.extend(["%s % 4d/%d" % (r[-1], collected_results[r[0]]["counts"]
                                      [0], numpy.sum(collected_results[r[0]]["counts"])) for r in rids])
            else:
                ytick_lbls_all.extend(["%d/%d" % (collected_results[r[0]]["counts"]
                                      [0], numpy.sum(collected_results[r[0]]["counts"])) for r in rids])

            suboffs.append(offset-1)
            sublbls.append("%s" % lbls_series[sss])

            offset += len(sdata)+2

        grid[li].set_xlim(x_min, x_max)
        ymin, ymax = (numpy.max(yticks_all)+1, numpy.min(yticks_all)-2)
        grid[li].set_ylim(ymin, ymax)
        grid[li].plot([PERFECT_VAL, PERFECT_VAL], [ymin, ymax],
                      ':', color="#999999", zorder=-1)
        grid[li].set_xlabel(SCORE_LBL)
        grid[li].set_yticks(yticks_all)
        grid[li].set_yticklabels(ytick_lbls_all)

        for si in range(len(suboffs)):
            grid[li].text(x_min+(x_max-x_min)/2., suboffs[si],
                          sublbls[si], horizontalalignment='center')
            grid[li].plot([x_min+.1, x_max-.1], [suboffs[si]+.35,
                          suboffs[si]+.35], "k", linewidth=0.1)

    plt.draw()
    plt.savefig("%s%s_box.pdf" % (PLT_REP, re.sub("\*", "", series_basis)))


if __name__ == "__main__":
    BASIS_REP = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    XPS_REP = BASIS_REP+"/xps/synthe/"
    PLT_REP = BASIS_REP+"/xps/"

    for series_basis in ["synthe_S", "synthe_V", "synthe_W", "synthe_U"]:
        draw_synthe_boxes(series_basis)

    series_basis = "synthe_*comb"
    draw_synthe_boxes_combs(series_basis)
