import numpy
import re
import sys
import pdb

SERIES = "vX"
if len(sys.argv) > 1:
    SERIES = sys.argv[1]

XPS_REP = "../xps/"
FILE_IN = XPS_REP+"%s" % SERIES

OUT_DT_STATS = XPS_REP+"table_datasets_stats.tex"
OUT_DT_AGG = XPS_REP+"table_datasets_agg.tex"
OUT_RES_ALL = XPS_REP+"table_results_all_long.tex"
OUT_RES_SHRT = XPS_REP+"table_results_short.tex"
OUT_RES_AGG = XPS_REP+"table_results_agg.tex"

SEP = " & "
EoL = " \\\\ "

FOOT = """ """

CMP_FIELD = "prc_cl"

DT_NAMES = {"3zap-0-rel": "\\dstTZap{0}",
            "3zap-1-rel": "\\dstTZap{1}",
            "bugzilla-0-rel-all": "\\dstBugz{0}",
            "bugzilla-1-rel-all": "\\dstBugz{1}",
            "sacha-18-abs-G1": "\\dstSachaAR{\\iAbs}",
            "sacha-18-abs": "\\dstSachaAR{\\iAbs}",
            "sacha-18-rel": "\\dstSachaAR{\\iRel}",
            "samba-auth-abs": "\\dstSamba{}"}

INTER_LBL = {"S": "$\\collS$", "V": "$\\collV$",
             "H": "$\\collH$", "V+H": "$\\collVH$", "F": "$\\collF$"}


def getDName(name):
    if name in DT_NAMES:
        return DT_NAMES[name]
    elif re.match("sacha-18-abs[I]?-G[0-9]+", name):
        grain = name.split("G")[-1]
        return "\\dstSachaG{%s}" % grain
    elif re.match("UbiqLog\-[0-9]+\-[FM]\-ISE\-abs", name):
        which = "-".join(name.split("-")[1:3])
        return "\\dstUbiSAbs{%s}" % which
    elif re.match("UbiqLog\-[0-9]+\-[FM]\-IS\-rel", name):
        which = "-".join(name.split("-")[1:3])
        return "\\dstUbiSRel{%s}" % which
    return re.sub("-ISE?-[a-z]+$", "", re.sub("^UbiqLog-", "", name))


def get_frac(Xorg, rid, inter, map_field_num, num=None, den=None):
    if num is None:
        return 0
    return (Xorg[rid, map_field_num["%s_%s" % (inter, num)]]/float(Xorg[rid, map_field_num["%s_%s" % (inter, den)]]))


def get_entry(Xorg, rid, inter, ifrmt, ifield, map_field_num, ids_min=[]):
    if ("%s_%s" % (inter, ifield)) in map_field_num:
        if ifield == CMP_FIELD and ii in ids_min:
            return ("$\\textbf{"+ifrmt+"}$") % Xorg[rid, map_field_num["%s_%s" % (inter, ifield)]]
        else:
            return ("$"+ifrmt+"$") % Xorg[rid, map_field_num["%s_%s" % (inter, ifield)]]
    else:
        if ifield == "split_nbs":
            return "/".join(["%d" % Xorg[rid, map_field_num["%s_%s" % (inter, iifield)]] for iifield in ["nb_simple", "nb_nested", "nb_concat", "nb_other"]])
        elif re.match("frac_", ifield):
            if ifield == "frac_nbO_>3":
                num, den = ("nbO_>3", "nb_patts")
            if ifield == "frac_residuals_cl":
                num, den = ("residuals_cl", "compressed_cl")
            if ifield == "frac_patt_cl":
                num, den = ("patt_cl", "compressed_cl")
            return "%.2f" % get_frac(Xorg, rid, inter, map_field_num, num, den)


def get_agg(Xorg, rids, inter, ifrmt, ifield, map_field_num, ops):
    vs = []
    if inter is None:
        kk = ifield
    else:
        kk = ("%s_%s" % (inter, ifield))
    if kk in map_field_num:
        vs = Xorg[rids, map_field_num[kk]]
    else:
        if re.match("frac_", ifield):
            if ifield == "frac_nbO_>3":
                num, den = ("nbO_>3", "nb_patts")
            if ifield == "frac_residuals_cl":
                num, den = ("residuals_cl", "compressed_cl")
            if ifield == "frac_patt_cl":
                num, den = ("patt_cl", "compressed_cl")
            vs = [get_frac(Xorg, rid, inter, map_field_num, num, den)
                  for rid in rids]
    if len(vs) > 0:
        if len(ops) == 1:
            return ("$"+ifrmt+"$") % ops[0][1](vs)
        else:
            return "["+", ".join([("$"+ifrmt+"$") % op[1](vs) for op in ops])+"]"
    return "-"


inters = ["S", "V", "H", "V+H", "F"]
head = None
rlabels = []
with open(FILE_IN) as fp:
    for line in fp:
        if head is None:
            head = line.strip().split()
        else:
            rlabels.append(line.split()[0])

Xorg = numpy.loadtxt(FILE_IN, usecols=range(1, len(head)), skiprows=1)
head.extend(["%s_delta_compressed_cl" % ii for ii in inters])
map_field_num = dict([(v, k-1) for (k, v) in enumerate(head)])
rlabels = [re.sub("_", "-", rlbl) for rlbl in rlabels]
rlabels = [re.sub("[_\-]v[0-9]?[0-9]?", "", rlbl) for rlbl in rlabels]


if Xorg.ndim == 1:
    Xorg = numpy.expand_dims(Xorg, axis=0)
Xorg = numpy.hstack([Xorg, numpy.vstack([(Xorg[:, map_field_num["S_prc_cl"]] -
                    Xorg[:, map_field_num["%s_prc_cl" % ii]]) for ii in inters]).T])


# DATA STATS
DATASET_FIELDS = [("%d", "size_O", "$\len{\seq}$"), ("%d", "size_T", "$\\tspan{\seq}$"), ("%d", "size_ABC", "$\\abs{\\ABC}$"),
                  ("%d", "evNbO_median", "$\\len{\\seq[\\alpha]}$"), (
                      "%d", "evNbO_max", "$\\len{\\seq[\\alpha]}$"),
                  ("", None, ""), ("%d", "F_original_cl", "$\\clEmpty$"),
                  ("%d", "runtime_cycles", "RT (s) cycles"), ("%d", "runtime_mining", "RT (s)")]
DATA_COL_FRMT = "@{\\hspace{1ex}}l@{\\hspace{4ex}}r@{\\hspace{2ex}}r@{\\hspace{2ex}}r@{\\hspace{2.5ex}}r@{\\hspace{1.5ex}}r@{\\hspace{2ex}}r@{\\hspace{2ex}}r@{\\hspace{2.5ex}}r@{\\hspace{1.5ex}}r@{\\hspace{1ex}}"
DATA_HEAD = SEP.join([""]+['$\\len{\\seq}$', '$\\tspan{\\seq}$', '$\\abs{\\ABC}$',
                     '\multicolumn{2}{c}{$\\len{\\seq[\\alpha]}$}', "",  '$\\clEmpty$', '\multicolumn{2}{c}{RT (s)}'])
DATA_HEAD_SEC = SEP.join(
    4*[""]+['$\\omed$', '$\\max$', '', '', 'cycles', 'overall'])
##########################
max_per_table = 7
indivs = []
rids = [r for r in range(len(rlabels)) if not re.match(
    "UbiqLog", rlabels[r]) and not re.match("sacha", rlabels[r])]
indivs.append({"title": "application log trace",
              "rids": rids, "ref": "traces"})

# ("UbiqLog.*IS[_\-]abs", "UbiqLog IS Abs", "UbiqLog-IS-abs"),
for mtch, tit, ref in [("sacha.*", "\\dstSacha{}", "sacha"), ("UbiqLog.*ISE", "\\dstUbiAR{\\iAbs}", "UbiqLog-ISE-abs"), ("UbiqLog.*IS[_\-]rel", "\\dstUbiAR{\\iRel}", "UbiqLog-IS-rel")]:
    rids = [r for r in range(len(rlabels)) if re.match(mtch, rlabels[r])]
    rids.sort(key=lambda x: Xorg[x, map_field_num["size_O"]])
    if ref == "sacha":
        skeys = dict([(rid, (int((rlabels[rid]+"G9999").split("G")[1]),
                     "2000" not in rlabels[rid], "absI" not in rlabels[rid])) for rid in rids])
        rids.sort(key=lambda x: skeys[x])

    indivs.append({"title": "%s" % tit,
                   "ref": "%s" % ref,
                   "rids": rids})

fo_dstats = open(OUT_DT_STATS, "w")
for group in indivs:
    str_table = ""
    str_table += "\\begin{table}\n"
    str_table += "\\caption{Statistics for %s sequences.}\n" % group["title"]
    str_table += "\\label{tab:data-stats-%s}\n" % group["ref"]
    str_table += "\\centering\n"
    str_table += "\\begin{tabular}{%s}\n" % DATA_COL_FRMT
    str_table += "\\toprule\n"
    str_table += DATA_HEAD + EoL + "\n"
    str_table += DATA_HEAD_SEC + EoL + "\n"
    str_table += "\\midrule\n"

    data_rows = []
    # rrs = sorted(group["rids"], key=lambda x: Xorg[x, map_field_num["size_O"]])
    rrs = group["rids"]
    for rid in rrs:
        dname = getDName(rlabels[rid])
        data_rows.append([dname])
        for (ifrmt, ifield, h) in DATASET_FIELDS:
            if ifield is not None:
                data_rows[-1].append(("$"+ifrmt+"$") %
                                     Xorg[rid, map_field_num[ifield]])
            else:
                data_rows[-1].append("")
        str_table += SEP.join(data_rows[-1]) + EoL + "\n"
        # if rid != rrs[-1]:
        #     str_table += "[1em]\n"
    str_table += "\\bottomrule\n"
    str_table += "\\end{tabular}\n"
    str_table += "\\end{table}\n"
    fo_dstats.write(str_table)
fo_dstats.write(FOOT)
fo_dstats.close()
###############################
##########################
max_per_table = 7
indivs = []
ops = [("min", numpy.min), ("median", numpy.median), ("max", numpy.max)]
# ("UbiqLog.*IS[_\-]abs", "UbiqLog IS Abs", "UbiqLog-IS-abs"),
for mtch, tit, ref in [("UbiqLog.*ISE", "\\dstUbiAR{\\iAbs}", "UbiqLog-ISE-abs"), ("UbiqLog.*IS[_\-]rel", "\\dstUbiAR{\\iRel}", "UbiqLog-IS-rel")]:
    rids = [r for r in range(len(rlabels)) if re.match(mtch, rlabels[r])]

    indivs.append({"title": "%s" % tit,
                   "ref": "%s" % ref,
                   "rids": rids})

fo_dstats = open(OUT_DT_AGG, "w")
for x in ["X"]:
    str_table = ""
    str_table += "\\begin{table}\n"
    str_table += "\\caption{Aggregated statistics for \\dstUbi{} sequences.}\n"
    str_table += "\\label{tab:data-stats-agg}\n"
    str_table += "\\centering\n"
    str_table += "\\begin{tabular}{%s}\n" % DATA_COL_FRMT
    str_table += "\\toprule\n"
    str_table += DATA_HEAD + EoL + "\n"
    str_table += DATA_HEAD_SEC + EoL + "\n"
    str_table += ("\\cmidrule{2-%d}" %
                  (len(DATASET_FIELDS)+1)) + EoL + "[-.5em]\n"

    data_rows = []
    # rrs = sorted(group["rids"], key=lambda x: Xorg[x, map_field_num["size_O"]])
    rrs = [0]
    for gi, group in enumerate(indivs):
        # re.sub("-ISE?-[a-z]+$", "", re.sub("^UbiqLog-", "", rlabels[rid]))
        dname = "%s (%d)" % (group["title"], len(group["rids"]))
        str_table += ("\\multicolumn{%d}{c}{%s}" %
                      (len(DATASET_FIELDS)+1, dname)) + EoL + "\n"
        str_table += "\\midrule\n"
        for op in ops:
            data_rows.append([op[0]])
            for (ifrmt, ifield, h) in DATASET_FIELDS:
                if ifield is not None:
                    data_rows[-1].append(get_agg(Xorg, group["rids"],
                                         None, ifrmt, ifield, map_field_num, [op]))
                    # data_rows[-1].append(("$"+ifrmt+"$") % Xorg[rid, map_field_num[ifield]])
                else:
                    data_rows[-1].append("")
            str_table += SEP.join(data_rows[-1]) + EoL + "\n"
        if gi < len(indivs)-1:
            str_table += "[.5em]\n"
    str_table += "\\bottomrule\n"
    str_table += "\\end{tabular}\n"
    str_table += "\\end{table}\n"
    fo_dstats.write(str_table)
fo_dstats.write(FOOT)
fo_dstats.close()
###############################

# LONG RESULTS
# wide tables with all results stats
inters = ["S", "V", "H", "V+H", "F"]
WIDE_FIELDS = [("%.2f", "prc_cl", "$\\prcCl{}$"), ("%d", "compressed_cl", "$\\clCC$"),
               ("%.2f", "frac_residuals_cl", "$\\ratioClR{}$"),
               ("%d", "nb_residuals",
                "$\\abs{\\resSet}$"), ("%d", "nb_patts", "$\\abs{\\ccycle}$"),
               ("%d", "nb_simple", "$\\nbS$"), ("%d", "nb_nested", "$\\nbV$"),
               ("%d", "nb_concat", "$\\nbH$"), ("%d", "nb_other", "$\\nbTD$"),
               ("%d", "frac_nbO_>3", "$\\nbOTC$"), ("%d", "nbO_median", "$\\nbOmed$"), ("%d", "nbO_max", "$\\nbOmax$")]  # ,
# ("%d", "r_median", "$r^m$"), ("%d", "r_max", "$r^+$")] ("%s", "split_nbs", "$\\#$"), ("%d", "nbO_mean", "$o^\sim$"),
nbC = len(WIDE_FIELDS)
WIDE_COL_FRMT = "@{}c@{\\hspace{3ex}}"+2*"r@{\\hspace{1ex}}"+"r@{\\hspace{3ex}}"+2 * \
    "r@{\\hspace{1ex}}" + 3*"c@{\\,/\\,}" + \
    "c@{\\hspace{3ex}}" + 2*"r@{\\hspace{1ex}}"+"r@{}"

# WIDE_COL_FRMT = "@{}c@{\\hspace{2ex}}"+ "@{\\hspace{1ex}}".join(["r" for i in WIDE_FIELDS])+"@{}"
tmp = [""]
tmp += ["%s" % df[-1] for df in WIDE_FIELDS[:2]]
tmp += ["%s" % df[-1] for df in WIDE_FIELDS[2:4]]
tmp += ["%s" % df[-1] for df in WIDE_FIELDS[4:nbC-7]]
tmp += ["\\parbox[b][1em][b]{.5cm}{\\centering %s}" %
        df[-1] for df in WIDE_FIELDS[nbC-7:-3]]
tmp += ["%s" % df[-1] for df in WIDE_FIELDS[-3:]]
WIDE_HEAD = SEP.join(tmp)
##########################

max_per_table = 7
indivs = []
rids = [r for r in range(len(rlabels)) if not re.match(
    "UbiqLog", rlabels[r]) and not re.match("sacha", rlabels[r])]
indivs.append({"title": "application log trace sequences",
              "rids": rids, "ref": "traces"})

# rids = [r for r in range(len(rlabels)) if re.match("sacha", rlabels[r])]
# indivs.append({"title": "\\dstSamba{} sequences", "rids": rids, "ref": "sa"})

# ("UbiqLog.*IS[_\-]abs", "UbiqLog IS Abs", "UbiqLog-IS-abs"),
for mtch, tit, ref in [("sacha", "\\dstSacha{}", "sacha"), ("UbiqLog.*ISE", "\\dstUbiAR{\\iAbs}", "UbiqLog-ISE-abs"), ("UbiqLog.*IS[_\-]rel", "\\dstUbiAR{\\iRel}", "UbiqLog-IS-rel")]:
    rids = [r for r in range(len(rlabels)) if re.match(mtch, rlabels[r])]
    rids.sort(key=lambda x: Xorg[x, map_field_num["size_O"]])
    if ref == "sacha":
        skeys = dict([(rid, (int((rlabels[rid]+"G9999").split("G")[1]),
                     "2000" not in rlabels[rid], "absI" not in rlabels[rid])) for rid in rids])
        rids.sort(key=lambda x: skeys[x])

    nb_t = int(numpy.ceil(len(rids)/float(max_per_table)))
    for i in range(nb_t):
        indivs.append({"title": "%s sequences (%d/%d)" % (tit, i+1, nb_t),
                       "ref": "%s_%d" % (ref, i+1),
                       "rids": rids[i*max_per_table:(i+1)*max_per_table]})
        if nb_t == 1:
            indivs[-1]["title"] = "%s" % tit

fo_wide = open(OUT_RES_ALL, "w")
for group in indivs:
    str_table = ""
    interss = inters
    # if "inter" in group:
    #     interss = group["inter"]

    str_table += "\\begin{table}\n"
    str_table += "\\caption{Detailed results for %s.}\n" % group["title"]
    str_table += "\\label{tab:res-long-%s}\n" % group["ref"]
    str_table += "\\centering\n"
    str_table += "\\begin{tabular}{%s}\n" % WIDE_COL_FRMT
    str_table += "\\toprule\n"
    str_table += WIDE_HEAD + EoL+"\n"
    str_table += ("\\cmidrule{2-%d}" %
                  (len(WIDE_FIELDS)+1)) + EoL + "[-.5em]\n"

    xps_rows = []
    # sorted(group["rids"], key=lambda x: Xorg[x, map_field_num["size_O"]])
    rrs = group["rids"]
    for rr, rid in enumerate(rrs):
        dname = getDName(rlabels[rid])
        str_table += ("\\multicolumn{%d}{c}{%s}" %
                      (len(WIDE_FIELDS)+1, dname)) + EoL + "\n"
        str_table += "\\midrule\n"
        xps_rows.append([])
        cmp_vals = [Xorg[rid, map_field_num["%s_%s" %
                                            (inter, CMP_FIELD)]] for inter in inters]
        min_val = numpy.min(cmp_vals)
        ids_min = numpy.where(cmp_vals == min_val)[0]
        for ii, inter in enumerate(interss):
            xps_rows[-1].append([INTER_LBL[inter]])
            for (ifrmt, ifield, h) in WIDE_FIELDS:
                xps_rows[-1][-1].append(get_entry(Xorg, rid,
                                        inter, ifrmt, ifield, map_field_num, ids_min))
            str_table += SEP.join(xps_rows[-1][-1]) + EoL + "\n"
        if rid != rrs[-1]:
            str_table += "[.5em]\n"
    str_table += "\\bottomrule\n"
    str_table += "\\end{tabular}\n"
    str_table += "\\end{table}\n"
    fo_wide.write(str_table)
fo_wide.write(FOOT)
fo_wide.close()
##############

# SHORT RESULTS
inters_sf = ["S", "F"]
inters_b0 = ["S", "V", "H", "V+H", "F"]
SHORT_FIELDS = [("%.2f", "prc_cl", "$\\prcCl{}$"),
                ("%.2f", "frac_residuals_cl", "$\\ratioClR{}$"),
                ("%d", "nb_simple", "$\\nbS$"), ("%d", "nb_nested", "$\\nbV$"),
                ("%d", "nb_concat", "$\\nbH$"), ("%d", "nb_other", "$\\nbTD$"),
                ("%d", "nbO_max", "$\\nbOmax$")]  # ,
# ("%d", "r_median", "$r^m$"), ("%d", "r_max", "$r^+$")] ("%s", "split_nbs", "$\\#$"), ("%d", "nbO_mean", "$o^\sim$"),
nbC = len(SHORT_FIELDS)
SHORT_COL_FRMT = "@{}c@{\\hspace{1ex}}r@{\\hspace{.7ex}}r@{\\hspace{.7ex}}" + \
    3*"c@{/}"+"c@{\\hspace{.4ex}}r@{}"
# SHORT_COL_FRMT = "@{}c@{\\hspace{2ex}}"+ "@{\\hspace{1ex}}".join(["r" for i in SHORT_FIELDS])+"@{}"

tmp = [""]
tmp += ["%s" % df[-1] for df in SHORT_FIELDS[:1]]
tmp += ["%s" % df[-1] for df in SHORT_FIELDS[1:2]]
tmp += ["\\parbox[b][1em][b]{.2cm}{\\centering %s}" %
        df[-1] for df in SHORT_FIELDS[2:-1]]
tmp += ["%s" % df[-1] for df in SHORT_FIELDS[-1:]]
SHORT_HEAD = SEP.join(tmp)
# SHORT_HEAD = SEP.join(["%s" % df[-1] for df in SHORT_FIELDS])
# ##########################

indivs = []

rids = [r for r in range(len(rlabels)) if re.match("3zap.*0", rlabels[r])]
rids += [r for r in range(len(rlabels)) if re.match("bug.*0", rlabels[r])]
rids += [r for r in range(len(rlabels)) if re.match("samba", rlabels[r])]
# rids = [r for r in range(len(rlabels)) if not re.match("UbiqLog", rlabels[r]) and not re.match("sa", rlabels[r])]
indivs.append({"title": "application log trace",
              "rids": rids, "ref": "traces"})

rids = [r for r in range(len(rlabels)) if re.match("3zap.*1", rlabels[r])]
rids += [r for r in range(len(rlabels)) if re.match("bug.*1", rlabels[r])]
rids += [r for r in range(len(rlabels)) if re.match("sacha.*G15$", rlabels[r])]
# rids = [r for r in range(len(rlabels)) if not re.match("UbiqLog", rlabels[r]) and re.match("sa", rlabels[r]) and not re.match("sacha.*G", rlabels[r])]
indivs.append({"title": "\\dstSacha{} and \\dstSamba{}",
              "rids": rids, "ref": "sa"})

fo_small = open(OUT_RES_SHRT, "w")
str_table = ""
str_table += "\\begin{table}\n"
# % group["title"]
str_table += "\\caption{Results for application separate event sequences.}\n"
str_table += "\\label{tab:res-short}\n"  # % group["ref"]
str_table += "\\centering\n"

inter_lbl = INTER_LBL
for ggi, group in enumerate(indivs):

    # if "inter" in group:
    #     interss = group["inter"]
    str_table += "\\begin{minipage}{.48\\textwidth}\n"
    str_table += "\\begin{tabular}{%s}\n" % SHORT_COL_FRMT
    str_table += "\\toprule\n"
    str_table += SHORT_HEAD + EoL+"\n"
    str_table += ("\\cmidrule{2-%d}" %
                  (len(SHORT_FIELDS)+1)) + EoL + "[-.5em]\n"

    xps_rows = []
    # sorted(group["rids"], key=lambda x: Xorg[x, map_field_num["size_O"]])
    rrs = group["rids"]
    for rr, rid in enumerate(rrs):
        interss = inters_sf
        # if re.match("bugzilla-0-rel-all", rlabels[rid]):
        # if re.match("3zap-0-rel-6000", rlabels[rid]):
        if True:  # re.match("3zap", rlabels[rid]):
            interss = inters
        dname = getDName(rlabels[rid])
        str_table += ("\\multicolumn{%d}{c}{%s}" %
                      (len(SHORT_FIELDS)+1, dname)) + EoL + "\n"
        str_table += "\\midrule\n"
        xps_rows.append([])
        cmp_vals = [Xorg[rid, map_field_num["%s_%s" %
                                            (inter, CMP_FIELD)]] for inter in inters]
        min_val = numpy.min(cmp_vals)
        ids_min = numpy.where(cmp_vals == min_val)[0]
        for ii, inter in enumerate(interss):
            xps_rows[-1].append([inter_lbl[inter]])
            for (ifrmt, ifield, h) in SHORT_FIELDS:
                xps_rows[-1][-1].append(get_entry(Xorg, rid,
                                        inter, ifrmt, ifield, map_field_num, ids_min))
            str_table += SEP.join(xps_rows[-1][-1]) + EoL + "\n"
        if rid != rrs[-1]:
            str_table += "[.5em]\n"
    str_table += "\\bottomrule\n"
    str_table += "\\end{tabular}\n"
    str_table += "\\end{minipage}\n"
    if ggi < len(indivs)-1:
        str_table += "\\hfill\n"
    # inter_lbl = dict([(k,"") for k in INTER_LBL])

str_table += "\\end{table}\n"
fo_small.write(str_table)
fo_small.write(FOOT)
fo_small.close()
##############


#######################
# AGGREGATED
# ops = [("min", numpy.min), ("med", numpy.median), ("max", numpy.max)]
ops = [("min", numpy.min), ("max", numpy.max)]
indivs = []
# ("UbiqLog.*IS[_\-]abs", "UbiqLog IS Abs", "UbiqLog-IS-abs"),
for mtch, tit, ref in [("UbiqLog.*ISE", "\\dstUbiAR{\\iAbs}", "UbiqLog-ISE-abs"), ("UbiqLog.*IS[_\-]rel", "\\dstUbiAR{\\iRel}", "UbiqLog-ISE-rel")]:
    # for mtch, tit, ref in [("UbiqLog.*ISE", "UbiqLog ISE Abs", "UbiqLog-ISE-abs"), ("UbiqLog.*IS[_\-]rel", "UbiqLog IS Rel", "UbiqLog-ISE-rel")]:
    rids = [r for r in range(len(rlabels)) if re.match(mtch, rlabels[r])]
    indivs.append({"title": "%s" % tit,
                   "ref": "%s" % ref,
                   "rids": rids})

fo_agg = open(OUT_RES_AGG, "w")
for x in ["X"]:
    str_table = ""
    interss = inters
    # if "inter" in group:
    #     interss = group["inter"]

    str_table += "\\begin{table}\n"
    str_table += "\\caption{Aggregated results for \\dstUbi{} sequences.}\n"
    str_table += "\\label{tab:res-agg}\n"
    str_table += "\\centering\n"
    str_table += "\\begin{tabular}{%s}\n" % SHORT_COL_FRMT
    str_table += "\\toprule\n"
    str_table += SHORT_HEAD + EoL+"\n"
    str_table += ("\\cmidrule{2-%d}" %
                  (len(SHORT_FIELDS)+1)) + EoL + "[-.5em]\n"

    xps_rows = []
    # group["rids"] #sorted(group["rids"], key=lambda x: Xorg[x, map_field_num["size_O"]])
    rrs = [0]
    for gi, group in enumerate(indivs):
        # re.sub("-ISE?-[a-z]+$", "", re.sub("^UbiqLog-", "", rlabels[rid]))
        dname = "%s (%d)" % (group["title"], len(group["rids"]))
        str_table += ("\\multicolumn{%d}{c}{%s}" %
                      (len(SHORT_FIELDS)+1, dname)) + EoL + "\n"
        str_table += "\\midrule\n"
        xps_rows.append([])
        for ii, inter in enumerate(inters):
            xps_rows[-1].append([INTER_LBL[inter]])
            for (ifrmt, ifield, h) in SHORT_FIELDS:
                xps_rows[-1][-1].append(get_agg(Xorg, group["rids"],
                                        inter, ifrmt, ifield, map_field_num, ops))
            str_table += SEP.join(xps_rows[-1][-1]) + EoL + "\n"
        if gi < len(indivs)-1:
            str_table += "[.5em]\n"
    str_table += "\\bottomrule\n"
    str_table += "\\end{tabular}\n"
    str_table += "\\end{table}\n"
    fo_agg.write(str_table)
fo_agg.write(FOOT)
fo_agg.close()
#######################
