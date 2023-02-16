import datetime
import glob
import re

FILENAME_DATA = "../data/SachaTrackSTMP/org/data_18-03-22_lagg200NL-XX.txt"
FILENAME_MAPIDS = "../data/SachaTrackSTMP/org/event_codes_out.txt"
XPS_REP = "../xps/sacha_text/"
MATCH_PATTS = "sacha_18_*_patts*.txt"

with open(FILENAME_DATA) as fp:
    tt = " ".join(fp.readline().split()[4:6])
abs_init_T = datetime.datetime.strptime(tt, "%Y-%m-%d %H:%M:%S")

codes_tmp = {}
with open(FILENAME_MAPIDS) as fp:
    for line in fp:
        parts = line.strip().split("\t")
        codes_tmp[parts[0]] = parts[1]
        # if parts[1] not in codes_tmp:
        #     codes_tmp[parts[1]] = set(enumerate(parts[2]))
        # else:
        #     codes_tmp[parts[1]].intersection_update(enumerate(parts[2]))

codes_abs = {}
codes_rel = {}
for code, cs in codes_tmp.items():
    codes_abs["%s_S" % code] = cs + "_START"
    codes_abs["%s_E" % code] = cs + "_END"
    codes_abs["%s_I" % code] = cs + "_INS"
    codes_rel["%s_S" % code] = cs
    codes_rel["%s" % code] = cs

for fn in glob.glob(XPS_REP + MATCH_PATTS):
    fo = open(re.sub("_patts", "_text-patts", fn), "w")
    if re.search("_absI?_", fn):
        is_abs = True
        codes = codes_abs
    else:
        is_abs = False
        codes = codes_rel
    tmp = re.search("_G(?P<grain>[0-9]+)_", fn)
    grain = 1.
    if tmp is not None:
        grain = float(tmp.group("grain"))
    with open(fn) as fp:
        for line in fp:
            if re.match("t0\=", line):
                suff = "X"
                parts = line.strip().split("\t")
                if is_abs:
                    pp = parts[0].split("=")
                    parts[0] = "t0=%s" % (
                            abs_init_T + datetime.timedelta(minutes=grain * int(pp[1])))

                Ptree = parts[1]
                for p in [pp for pp in re.finditer("([^() ]+)", parts[1])][::-1]:
                    if Ptree[p.start():p.end()] in codes:
                        Ptree = "%s%s%s" % (
                            Ptree[:p.start()], codes[Ptree[p.start():p.end()]], Ptree[p.end():])
                    elif is_abs:
                        tmp = re.match(
                            "(?P<bef>\[?[dp]\=)(?P<time>[0-9]+)\]", Ptree[p.start():p.end()])
                        if tmp is not None and int(tmp.group("time")) != 0:
                            if grain * int(tmp.group("time")) > 29 * 24 * 60:
                                suff = "M"
                            elif grain * int(tmp.group("time")) > 6.5 * 24 * 60 and suff in ["D", "X"]:
                                suff = "W"
                            elif grain * int(tmp.group("time")) > 22 * 60 and suff == "X":
                                suff = "D"

                            td = "%s" % datetime.timedelta(
                                minutes=grain * int(tmp.group("time")))
                            tt = re.sub(":00$", "", re.sub(
                                " days?, ", "D,", td))
                            Ptree = "%s%s%s]%s" % (
                                Ptree[:p.start()], tmp.group("bef"), tt, Ptree[p.end():])
                parts[1] = Ptree
                fo.write("\t".join(parts) + ("\t%s\n" % suff))
            else:
                fo.write(line)
    fo.close()

# sed 's/\([ (]\)\([^ (]*\)_START/\1\\activityStart{\2}/g;s/\([ (]\)\([^ (]*\)_END/\1\\activityEnd{\2}/g;s/\([ (]\)\(
# [^ (]*\)_INS/\1\\activityIns{\2}/g;s/Code length:\([0-9]*\)\.[0-9]*/\1/g;s/sum[^=]*=//g;s:Occs (\([0-9]*\)/[
# 0-9]*).*$:\1:g;s/t0=/\& $/;s/\t/$ \& $/g;s/$/$ \\\\/' select_patts.txt

# sed 's/\([ (]\)\([^ (]*\)_START/\1\\activityStart{\2}/g;s/\([ (]\)\([^ (]*\)_END/\1\\activityEnd{\2}/g;s/\([ (]\)\(
# [^ (]*\)_INS/\1\\activityIns{\2}/g;s/Code length:\([0-9]*\)\.[0-9]*/\1/g;s/sum[^=]*=//g;s:Occs (\([0-9]*\)/[
# 0-9]*).*$:\1:g;s/t0=/\& $/;s/\t/$ \& $/g;s/$/$ \\\\/' select_patts_v01.txt | sed 's/\[r=\([0-9]*\) p=\([0-9:D,
# ]*\)\]/\\BinfoRPT{\1}{\2}/g;s/\[d=\([0-9:D,]*\)\]/\\BinfoDT{\1}/g' > select_patts_v01.tex
