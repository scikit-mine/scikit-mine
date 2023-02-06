import numpy
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import pdb

SERIES = "vX"
if len(sys.argv) > 1:
    SERIES = sys.argv[1]

XPS_REP = "../xps/"
FILE_IN = XPS_REP + "%s" % SERIES
FILE_IN_SACHA = XPS_REP + "%s" % SERIES

BASIS_OUT = XPS_REP + "fig"

CMAP_CMP = "rainbow"
CMAP_NBC = "binary"

CC = "S_nbOdup_max"
CX = "S_nb_cands"

inters = ["S", "V", "H", "V+H", "F"]

colors_all = ["r", "b", "g", "c", "k"]
colors_all = ["#332288", "#88CCEE", "#44AA99", "#117733",
              "#DDCC77", "#CC6677", "#AA4499", "#999933"]
colors_all = ["#332288", "#88CCEE", "#44AA99", "#DDCC77", "#CC6677"]

DT_NAMES_SHORT = {"3zap-0-rel": "3zap-0",
                  "3zap-1-rel": "3zap-1",
                  "bugzilla-0-rel-all": "bugzilla-0",
                  "bugzilla-1-rel-all": "bugzilla-1",
                  "sacha-18-absI-G1": "sacha-abs",
                  "sacha-18-abs": "sacha-abs",
                  "samba-auth-abs": "samba"}

DT_NAMES = {"sacha-18-rel": "sacha-rel"}
DT_NAMES.update(DT_NAMES_SHORT)


def getDName(name):
    name = re.sub("-v[0-9]*$", "", re.sub("_", "-", name))
    if name in DT_NAMES:
        return DT_NAMES[name]
    elif re.match("abs[I]?-G[0-9]+", name):
        grain = name.split("G")[-1]
        return "G%s" % grain
    elif re.match("UbiqLog\-[0-9]+\-[FM]\-ISE\-abs", name):
        which = "-".join(name.split("-")[1:3])
        return "%s" % which
    elif re.match("UbiqLog\-[0-9]+\-[FM]\-IS\-rel", name):
        which = "-".join(name.split("-")[1:3])
        return "%s" % which
    return re.sub("-ISE?-[a-z]+$", "", re.sub("^UL-", "", name))


head = None
rlabels = []
with open(FILE_IN) as fp:
    for line in fp:
        if head is None:
            head = line.strip().split()
        else:
            rlabels.append(line.split()[0])

# SUMMARY_TIME_PLOT
###############
if True:
    map_field_num = dict([(v, k - 1) for (k, v) in enumerate(head)])
    X = numpy.loadtxt(FILE_IN, usecols=range(1, len(head)), skiprows=1)
    Urids = [r for r in range(len(rlabels))
             if re.match("Ubi.*abs", rlabels[r])]
    rids = [r for r in range(len(rlabels)) if not re.match("Ubi", rlabels[r]) and (
            not re.match("sacha.*G", rlabels[r]) or re.match("sacha.*G15", rlabels[r])) and not re.search("_1_rel",
                                                                                                          rlabels[
                                                                                                              r])]
    Arids = range(len(rlabels))
    if X.ndim == 1:
        X = numpy.expand_dims(X, axis=0)

    max_CX = numpy.max(X[Arids, map_field_num[CX]])

    max_CC = 4.2  # numpy.log10(numpy.max(X[Arids,map_field_num[CC]]))
    min_CC = .8  # numpy.log10(numpy.min(X[Arids,map_field_num[CC]]))
    font = {'size': 22}
    matplotlib.rc('font', **font)
    msize = 120
    mmsize = 60
    sc_t = {"m": "min"}

    for sc in ["m", "h"]:
        plt.figure(figsize=(8, 5))

        # plt.scatter(X[Urids,map_field_num["size_O"]], X[Urids,map_field_num["runtime_mining"]], c=X[Urids,map_field_num["F_prc_cl"]], s=msize, marker='o', vmin=0, vmax=100, cmap=CMAP_CMP, zorder=20)
        # plt.scatter(X[rids,map_field_num["size_O"]], X[rids,map_field_num["runtime_mining"]], c=X[rids,map_field_num["F_prc_cl"]], s=msize, marker="s", vmin=0, vmax=100, cmap=CMAP_CMP, zorder=20)

        # plt.scatter(X[Urids,map_field_num["size_O"]], X[Urids,map_field_num["runtime_mining"]], c=numpy.log10(X[Urids,map_field_num[CC]]), s=mmsize+80*X[Urids,map_field_num[CX]]/max_CX, marker='o', cmap=CMAP_NBC, zorder=20, vmin=min_CC, vmax=max_CC)
        # plt.scatter(X[rids,map_field_num["size_O"]], X[rids,map_field_num["runtime_mining"]], c=numpy.log10(X[rids,map_field_num[CC]]), s=mmsize+80*X[rids,map_field_num[CX]]/max_CX, marker='s', cmap=CMAP_NBC, zorder=20, vmin=min_CC, vmax=max_CC)

        plt.plot(X[Urids, map_field_num["size_O"]],
                 X[Urids, map_field_num["runtime_mining"]], "ko")
        plt.plot(X[rids, map_field_num["size_O"]],
                 X[rids, map_field_num["runtime_mining"]], "ko")

        if sc == "h":
            plt.plot([-1000, -1000, 45000, 45000],
                     [-100, 3700, 3700, -100], "--", color="darkgray")
            plt.xlim([-1000, 195000])
            plt.ylim([-100, 10.5 * 3600])
            ycmin, ycmax = plt.ylim()
            plt.yticks(numpy.arange(0, ycmax, 3600), [
                       "% 4d" % v for v in numpy.arange(0, ycmax / 3600)])
        else:
            plt.xlim([-1000, 45000])
            plt.ylim([-100, 3700])
            ycmin, ycmax = plt.ylim()
            plt.yticks(numpy.arange(0, ycmax, 600), [
                       "% 4d" % v for v in numpy.arange(0, ycmax / 60, 10)])

        plt.ylabel("RT (%s)" % {"m": "min"}.get(sc, sc))
        plt.xlabel("|S|")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.95)
        plt.draw()
        det = "all"
        plt.savefig("%s_times(%s)_%s.pdf" % (BASIS_OUT, sc, det))
        plt.clf()
###############


# DETAILED_TIME_PLOT
###############
if True:
    map_field_num = dict([(v, k - 1) for (k, v) in enumerate(head)])
    X = numpy.loadtxt(FILE_IN, usecols=range(1, len(head)), skiprows=1)
    Urids = [r for r in range(len(rlabels)) if re.match("Ubi", rlabels[r])]
    rids = [r for r in range(len(rlabels)) if not re.match("Ubi", rlabels[r])]
    Arids = range(len(rlabels))
    if X.ndim == 1:
        X = numpy.expand_dims(X, axis=0)
    max_CX = numpy.max(X[Arids, map_field_num[CX]])
    max_CC = 4.2  # numpy.log10(numpy.max(X[Arids,map_field_num[CC]]))
    min_CC = .8  # numpy.log10(numpy.min(X[Arids,map_field_num[CC]]))
    font = {'size': 18}
    matplotlib.rc('font', **font)
    msize = 120
    mmsize = 60
    sc_t = {"m": "min"}

    for sc in ["s", "m", "h", "l"]:

        if sc == "l":
            plt.figure(figsize=(9, 6))
        else:
            plt.figure(figsize=(6, 4))
        for rid in range(len(rlabels)):
            plt.plot([X[rid, map_field_num["size_O"]], X[rid, map_field_num["size_O"]]],
                     [X[rid, map_field_num["runtime_combine"]],
                         X[rid, map_field_num["runtime_mining"]]], ":k",
                     zorder=5)

        plt.scatter(X[Urids, map_field_num["size_O"]], X[Urids, map_field_num["runtime_mining"]],
                    c=X[Urids, map_field_num["F_prc_cl"]], s=msize, marker='o', vmin=0, vmax=100, cmap=CMAP_CMP,
                    zorder=20)
        plt.scatter(X[rids, map_field_num["size_O"]], X[rids, map_field_num["runtime_mining"]],
                    c=X[rids, map_field_num["F_prc_cl"]], s=msize, marker="s", vmin=0, vmax=100, cmap=CMAP_CMP,
                    zorder=20)

        if sc == "l":
            cb = plt.colorbar(orientation="horizontal")
            cb.set_label("$\%\mathit{L}$")

        plt.scatter(X[Urids, map_field_num["size_O"]], X[Urids, map_field_num["runtime_combine"]],
                    c=numpy.log10(X[Urids, map_field_num[CC]]), s=mmsize + 80 * X[Urids, map_field_num[CX]] / max_CX,
                    marker='^', cmap=CMAP_NBC, zorder=10, vmin=min_CC, vmax=max_CC)
        plt.scatter(X[rids, map_field_num["size_O"]], X[rids, map_field_num["runtime_combine"]],
                    c=numpy.log10(X[rids, map_field_num[CC]]), s=mmsize + 80 * X[rids, map_field_num[CX]] / max_CX,
                    marker="^", cmap=CMAP_NBC, zorder=10, vmin=min_CC, vmax=max_CC)

        if sc == "l":
            # cb = plt.colorbar(orientation="vertical")
            cb = plt.colorbar(orientation="horizontal")
            cb.set_ticks([1, 2, 3, 4])
            cb.set_ticklabels(['$10$', '$10^2$', '$10^3$', '$10^4$'])
            cb.set_label("$c^+$")

        if sc == "h":
            plt.plot([-1000, -1000, 22000, 22000], [-100, 1300,
                     1300, -100], "--", color="darkgray", zorder=30)
            plt.plot([-1000, -1000, 45000, 45000], [-100, 3700,
                     3700, -100], "--", color="darkgray", zorder=30)

            plt.xlim([-1000, 195000])
            plt.ylim([-100, 10.5 * 3600])
            ycmin, ycmax = plt.ylim()
            plt.yticks(numpy.arange(0, ycmax, 3600), [
                       "% 4d" % v for v in numpy.arange(0, ycmax / 3600)])

        elif sc == "m":
            plt.plot([-1000, -1000, 22000, 22000], [-100, 1300,
                     1300, -100], "--", color="darkgray", zorder=30)
            plt.xlim([-1000, 45000])
            plt.ylim([-100, 3700])
            ycmin, ycmax = plt.ylim()
            plt.yticks(numpy.arange(0, ycmax, 600), [
                       "% 4d" % v for v in numpy.arange(0, ycmax / 60, 10)])

        else:
            plt.xlim([-1000, 22000])
            plt.ylim([-100, 1300])

        if sc != "l":
            plt.ylabel("RT (%s)" % {"m": "min"}.get(sc, sc))
            plt.xlabel("|S|")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.95)
        plt.draw()
        det = "details"
        plt.savefig("%s_times(%s)_%s.pdf" % (BASIS_OUT, sc, det))
        plt.clf()


# =================================================================================


ucols = [ci for ci in range(len(head)) if re.search("prc_cl", head[ci])]
map_inter_to_cid = dict([(v, k) for (k, v) in enumerate(
    [head[ci].split("_")[0] for ci in ucols])])
Xorg = numpy.loadtxt(FILE_IN, usecols=ucols, skiprows=1)

scols = [ci for ci in range(len(head)) if re.search("size_O", head[ci])]
Sorg = numpy.loadtxt(FILE_IN, usecols=scols, skiprows=1)
bar_labels = [re.sub("_prc_cl", "", head[ui]) for ui in ucols]

groups = []
rids = [r for r in range(len(rlabels)) if re.match("Ubi.*ISE", rlabels[r])]
rids.sort(key=lambda x: -Sorg[x])
groups.append({"title": "UbiqLog ISE Abs", "inter": ["S", "F"], "rids": rids,
               "rlabels": [rlabels[r] for r in rids]})

groups.append({"title": "UbiqLog ISE Abs 2-2", "inter": ["S", "F"], "rids": rids[int(len(rids) / 2):],
               "rlabels": [rlabels[r] for r in rids[int(len(rids) / 2):]]})
groups.append({"title": "UbiqLog ISE Abs 1-2", "inter": ["S", "F"], "rids": rids[:int(len(rids) / 2)],
               "rlabels": [rlabels[r] for r in rids[:int(len(rids) / 2)]]})

rids = [r for r in range(len(rlabels)) if re.match("Ubi.*IS_rel", rlabels[r])]
rids.sort(key=lambda x: -Sorg[x])
groups.append({"title": "UbiqLog IS Rel", "inter": ["S", "F"], "rids": rids,
               "rlabels": [rlabels[r] for r in rids]})
rids = [r for r in range(len(rlabels)) if not re.match(
    "Ubi.*", rlabels[r]) and not re.match("sacha", rlabels[r])][::-1]
groups.append({"title": "Other", "rids": rids,
               "rlabels": [rlabels[r] for r in rids]})

for group in groups:
    font = {'size': 12}
    matplotlib.rc('font', **font)
    if re.search("-2$", group["title"]):
        font = {'size': 20}
        matplotlib.rc('font', **font)

    if Xorg.ndim == 1:
        Xorg = numpy.expand_dims(Xorg, axis=0)
    X = Xorg[group["rids"], :]

    colors = colors_all
    blabels = bar_labels
    if "inter" in group:
        cids = [map_inter_to_cid[inter] for inter in group["inter"]]
    else:
        cids = [map_inter_to_cid[inter] for inter in inters]
    X = X[:, cids]
    colors = [colors_all[c] for c in cids]
    blabels = ["$\\mathcal{C}_{%s}$" % bar_labels[c] for c in cids]
    rlabels = ["%s" % getDName(rll) for rll in group["rlabels"]]

    plt.figure(figsize=(8, 5))
    for i in range(X.shape[1]):
        plt.barh(numpy.arange(X.shape[0]) + i * .8 / X.shape[1] - .8 / 2, X[:, i], .8 * .8 / X.shape[1],
                 color=colors[i])
        plt.barh(-2, 1, 1, color=colors[i], label=blabels[i])
    if re.search("-2$", group["title"]):
        plt.yticks(range(len(group["rlabels"])), rlabels, fontsize=18)
    elif "Other" == group["title"]:
        plt.yticks(range(len(group["rlabels"])), rlabels, fontsize=12)
    else:
        plt.yticks(range(len(group["rlabels"])), rlabels, fontsize=10)
    plt.xlabel("$\\%\\mathit{L}$")
    plt.xlim([0, 100])
    plt.ylim([-1, X.shape[0]])
    if re.search("2-2", group["title"]) is None:
        plt.legend(loc=4, frameon=False)
    if re.search("-2$", group["title"]):
        plt.subplots_adjust(bottom=0.14, top=.92)
    plt.draw()
    plt.savefig("%s_prcCL_%s.pdf" %
                (BASIS_OUT, re.sub(" ", "", group["title"])))

# SACHA
head = None
rlabels = []
with open(FILE_IN_SACHA) as fp:
    for line in fp:
        if head is None:
            head = line.strip().split()
        else:
            rlabels.append(line.split()[0])

ucols = [ci for ci in range(len(head)) if re.search("prc_cl", head[ci])]
map_inter_to_cid = dict([(v, k) for (k, v) in enumerate(
    [head[ci].split("_")[0] for ci in ucols])])
Xorg = numpy.loadtxt(FILE_IN_SACHA, usecols=ucols, skiprows=1)

bar_labels = [re.sub("_prc_cl", "", head[ui]) for ui in ucols]

groups = []
rids = [r for r in range(len(rlabels)) if re.match("sacha", rlabels[r])]
skeys = dict([(rid, (
    int((rlabels[rid] + "G9999").split("G")[1].split("_")[0]), "2000" not in rlabels[rid], "absI" not in rlabels[rid]))
    for
    rid in rids])
rids.sort(key=lambda x: skeys[x], reverse=True)
groups.append({"title": "Sacha", "inter": ["S", "F"], "rids": rids,
               "rlabels": [re.sub("2000", "S", re.sub("_v[0-9]*", "", re.sub("sacha_18_", "", rlabels[r]))) for r in
                           rids]})

font = {'size': 12}
matplotlib.rc('font', **font)

for group in groups:

    if Xorg.ndim == 1:
        Xorg = numpy.expand_dims(Xorg, axis=0)
    X = Xorg[group["rids"], :]
    colors = colors_all
    blabels = bar_labels
    if "inter" in group:
        cids = [map_inter_to_cid[inter] for inter in group["inter"]]
    else:
        cids = [map_inter_to_cid[inter] for inter in inters]
    X = X[:, cids]
    colors = [colors_all[c] for c in cids]
    blabels = ["$\\mathcal{C}_{%s}$" % bar_labels[c] for c in cids]
    rlabels = ["%s" % getDName(rll) for rll in group["rlabels"]]

    plt.figure(figsize=(8, 2))
    for i in range(X.shape[1]):
        plt.barh(numpy.arange(X.shape[0]) + i * .8 / X.shape[1] - .8 / 2, X[:, i], .8 * .8 / X.shape[1],
                 color=colors[i])
        plt.barh(-2, 1, 1, color=colors[i], label=blabels[i])
    plt.yticks(range(len(group["rlabels"])), rlabels, fontsize=12)
    plt.xlabel("$\\%\\mathit{L}$")
    plt.xlim([0, 100])
    plt.ylim([-1, X.shape[0]])
    plt.legend(loc=4, frameon=False)
    plt.subplots_adjust(bottom=0.32, top=0.95)
    plt.draw()
    plt.savefig("%s_prcCL_%s.pdf" %
                (BASIS_OUT, re.sub(" ", "", group["title"])))
