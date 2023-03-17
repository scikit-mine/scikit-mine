import datetime
import itertools

import numpy

from .candidate import Candidate
from .candidate_pool import CandidatePool
from .class_patterns import computePeriod, cost_one, sortPids
from .class_patterns import prop_map
from .data_sequence import DataSequence
from .extract_cycles import compute_cycles_dyn, extract_cycles_fold
from .pattern import Pattern
from .pattern_collection import PatternCollection

numpy.set_printoptions(suppress=True)

OFFSETS_T = [0, 1, -1]
TOP_KEACH = 5
CHECK_HORDER = True


def bronKerbosch3Plus(graph, collect, P, R=None, X=None):
    """
    Algorithm for finding maximal cliques in an undirected graph.
    A clique is a subset of vertices of an undirected graph such that every two distinct vertices  in the clique
    are adjacent. A maximal clique is a clique that cannot be extended by including one more adjacent vertex.

    Parameters
    ----------
    graph : dict
        A dictionary where each key (vertex) contains a vertex set to which the key is linked
    collect : list
        A set list where each set is a clique that contains the vertices
    P : set
    R : set
    X : set

    Returns
    -------
    None
        The list of cliques is in the variable collect
    """
    if X is None:
        X = set()
    if R is None:
        R = set()
    if len(P) == 0 and len(X) == 0:
        if len(R) > 2:
            collect.append(R)
    else:
        lP = list(P)
        for v in lP:
            bronKerbosch3Plus(graph, collect, P.intersection(graph[v]), R.union([v]), X.intersection(graph[v]))
            P.remove(v)
            X.add(v)


def prepare_tree_nested(cand, prds, lens):
    """
    Generate the tree of a nested candidate

    Parameters
    ----------
    cand
    prds : list
        Period values
    lens : list
        Repetition numbers
    Returns
    -------
    dict
        The nested tree
    """
    P = cand.getPattern()
    if P is None:
        depth = 1
    else:
        depth = len(P.getCyclePs())

    tree = {}
    for i in range(len(prds) - depth + 1):
        tree[i] = {"p": prds[i], "r": lens[i],
                   "children": [(i + 1, 0)], "parent": i - 1}
    tree[0]["parent"] = None

    if P is None:
        tree[i + 1] = {"event": cand.getEvent(), "parent": i}
    else:
        nodes, offset, nmap = cand.getTranslatedPNodes(i)
        ch = nodes.pop(nmap[0])["children"]
        for c in ch:
            nodes[c[0]]["parent"] = i
        tree[i]["children"] = ch
        tree.update(nodes)
    return tree


def prepare_candidate_nested(cp_det, cmplx_candidates):  # pragma : no cover
    """
    FIXME : to be explained
    Prepare a new nested candidate

    Parameters
    ----------
    cp_det
    cmplx_candidates

    Returns
    -------
    Candidate

    """
    idxs = cp_det[-1]
    pr_key = cp_det[1]
    tmp = pr_key.split("_")
    prds = list(map(int, tmp[0].split("+")))
    lens = list(map(int, tmp[1].split("*")))[::-1]

    occs = []
    for idx in idxs:
        occs.extend(cmplx_candidates[idx].getOccs())

    if cmplx_candidates[idxs[0]].getPattern() is None:
        depth = 1
    else:
        depth = len(cmplx_candidates[idxs[0]].getPattern().getCyclePs())

    list_reps = list(itertools.product(*[range(l) for l in lens[:-depth]]))
    map_reps = dict([(v, k) for (k, v) in enumerate(list_reps)])
    t00, Es = (None, [])
    for pi, pp in enumerate(list_reps):
        t0i = cmplx_candidates[idxs[pi]].getT0()
        Ei = cmplx_candidates[idxs[pi]].getE()
        if pi != 0:
            copy_pp = list(pp)
            i = len(pp) - 1
            while pp[i] == 0:
                i -= 1
            copy_pp[i] -= 1
            t0prec = cmplx_candidates[idxs[map_reps[tuple(copy_pp)]]].getT0()
            Es.append((t0i - t0prec) - prds[i])
        Es.extend(Ei)
    tree = prepare_tree_nested(cmplx_candidates[idxs[0]], prds, lens)
    p = Pattern(tree)
    return Candidate(-1, p, occs, Es)


def prepare_candidate_concats(cands, p0, r0, first_rs):
    """
    FIXME : to be explained

    Parameters
    ----------
    cands
    p0
    r0
    first_rs

    Returns
    -------

    """
    tree = {0: {"p": p0, "r": r0, "children": [], "parent": None}}
    offset = 0
    Pblks_occs = []
    Pblks_errs = []
    for ci, cand in enumerate(cands):

        Pblks_occs.append(cand.getBlocksO(first_rs[ci]))
        Pblks_errs.append(cand.getBlocksE(first_rs[ci]))

        nodes, offset, nmap = cand.getTranslatedPNodes(offset)
        offset -= 1
        ch = nodes.pop(nmap[0])["children"]
        if ci > 0:
            ch[0] = (ch[0][0], Pblks_occs[ci][0][0] - Pblks_occs[ci - 1][0][0])
        tree[0]["children"].extend(ch)
        for c in ch:
            nodes[c[0]]["parent"] = 0
        tree.update(nodes)

    ds = [c[1] for c in tree[0]["children"]]
    O = []
    E = []
    for ri in range(r0):
        for ei in range(len(Pblks_occs)):
            O.extend(Pblks_occs[ei][ri])
            if ei == 0:
                if ri != 0:
                    E.append((Pblks_occs[ei][ri][0] -
                              Pblks_occs[ei][ri - 1][0]) - p0)
            else:
                E.append((Pblks_occs[ei][ri][0] -
                          Pblks_occs[ei - 1][ri][0]) - ds[ei])
            E.extend(Pblks_errs[ei][ri])
    p = Pattern(tree)
    return Candidate(-1, p, O, E)


def mine_cycles_alpha(occs, alpha, data_details, costOne, max_p=None):
    """
    FIXME : to be explained
    Mine initial candidates

    Parameters
    ----------
    occs
    alpha
    data_details
    costOne
    max_p

    Returns
    -------

    """
    return extract_cycles_alpha(occs, alpha, data_details, costOne, max_p)


def extract_cycles_alpha(occs, alpha, data_details, costOne, max_p=None):
    """
    FIXME : to be explained

    Parameters
    ----------
    occs
    alpha
    data_details
    costOne
    max_p

    Returns
    -------

    """
    bound_dE = numpy.log2(data_details["deltaT"] + 1) - 2

    dyn_cycles = compute_cycles_dyn(occs, alpha, data_details, residuals=False)
    drop_occs = set()
    for dc in dyn_cycles:
        if len(dc["occs"]) > 10:
            drop_occs.update(dc["occs"][5:-5])
    if len(drop_occs) > 0:
        # FIX (  numpy.array  )  to not send a list to extract_cycles_fold()
        tmp_occs = numpy.array(sorted(set(occs).difference(drop_occs)))

    else:
        tmp_occs = occs

    chains, triples_tmp = extract_cycles_fold(
        tmp_occs, alpha, data_details, bound_dE, costOne, costOne, max_p)
    triples = [{"alpha": alpha, "occs": [tmp_occs[tt] for tt in t[-1]],
                "p": computePeriod([tmp_occs[tt] for tt in t[-1]]), "cost": t[-2]} for t in triples_tmp]

    cycles = [Candidate(-1, c)
              for c in merge_cycle_lists([dyn_cycles, chains, triples])]

    selected_ids = filter_candidates_topKeach(cycles, k=TOP_KEACH)
    return [cycles[s] for s in selected_ids]


def merge_cycle_lists(cyclesL):
    """
    Merge two cycle lists if they have the same t0, number of occurrences when reconstructed and period.
    The one with the lowest cost is kept.

    Parameters
    ----------
    cyclesL : list
        A list of cycle lists for each cycle

    Returns
    -------
        list
    The list of merged cycles
    """
    keys = []
    for ci, cycles in enumerate(cyclesL):
        # ((t0, nb occs, period), cycle_index, cyclesL_index
        keys.extend([((kk["occs"][0], len(kk["occs"]), kk["p"]), ki, ci)
                     for ki, kk in enumerate(cycles)])
    keys.sort()  # sort by ascending t0
    cycles = []
    if len(keys) > 0:
        cycles = [cyclesL[keys[0][2]][keys[0][1]]]  # get the cycle associated with t0
        cycles[-1]["source"] = (keys[0][2], keys[0][1])

    for i in range(1, len(keys)):
        if keys[i][0] != keys[i - 1][0]:  # if the first tuple is different between keys i and i-1
            cycles.append(cyclesL[keys[i][2]][keys[i][1]])
            cycles[-1]["source"] = (keys[i][2], keys[i][1])
        else:  # if they are the same, we merge the cycles
            if cyclesL[keys[i][2]][keys[i][1]]["cost"] < cycles[-1]["cost"]:  # if the cost of cycle i is less than
                # the last added in cycles
                cycles[-1] = cyclesL[keys[i][2]][keys[i][1]]
                cycles[-1]["source"] = (keys[i][2], keys[i][1])
    return cycles


def run_combine_vertical(cpool, data_details, nkey="H"):
    """
    FIXME : to be explained
    Combine candidates vertically

    Parameters
    ----------
    cpool
    data_details
    nkey

    Returns
    -------

    """
    minorKeys = cpool.getNewMinorKeys(nkey)
    candidates = []
    for mk in minorKeys:
        if len(cpool.getCidsForMinorK(mk)) >= 3:
            # only for simple events
            candidates.extend(run_combine_vertical_cands(
                cpool, mk, data_details))

    if len(candidates) > 0:
        selected_ids = filter_candidates_topKeach(candidates, k=TOP_KEACH)
        return [candidates[s] for s in selected_ids]
    return []


def run_combine_vertical_cands(cpool, mk, data_details):
    """
    FIXME : to be explained

    Parameters
    ----------
    cpool
    mk
    data_details

    Returns
    -------

    """
    cmplx_candidatesX = [cpool.getCandidate(
        cid) for cid in cpool.getCidsForMinorK(mk)]
    nested, covered = nest_cmplx(cmplx_candidatesX, data_details)

    if len(nested) > 0:
        selected_ids = filter_candidates_topKeach(nested, k=TOP_KEACH)
        return [nested[s] for s in selected_ids]
    return []


def get_top_p(occ_ordc):
    """
    FIXME : to be explained

    Parameters
    ----------
    occ_ordc

    Returns
    -------

    """
    top1, top2, topN = (0, 1, -1)
    if top2 + 1 < len(occ_ordc) and occ_ordc[top2][0] == occ_ordc[top2 + 1][0]:
        topN = top2 + 1
        while topN + 1 < len(occ_ordc) and occ_ordc[topN][0] == occ_ordc[topN + 1][0]:
            topN += 1
        if numpy.abs(occ_ordc[top1][1] - occ_ordc[topN][1]) > numpy.abs(occ_ordc[top1][1] - occ_ordc[top2][1]):
            (top2, topN) = (topN, top2)
    return top1, top2, topN


def nest_cmplx(cmplx_candidates, data_details):
    """
    FIXME : to be explained

    Parameters
    ----------
    cmplx_candidates
    data_details

    Returns
    -------

    """
    map_cmplx_pos = {}
    for cki, c in enumerate(cmplx_candidates):
        if c.getPattern() is not None:
            p_str = "+".join(["%d" %
                              ppp for ppp in c.getPattern().getCyclePs()])
            r_str = "*".join(["%d" %
                              ppp for ppp in c.getPattern().getCycleRs()[::-1]])
            pr_key = p_str + "_" + r_str
        else:
            pr_key = "%d_%d" % (c.getMajorP(), c.getMajorR())
        if pr_key not in map_cmplx_pos:
            map_cmplx_pos[pr_key] = []
        map_cmplx_pos[pr_key].append((c.getT0(), [cki]))

    keep = []
    while len(map_cmplx_pos) > 0:
        map_prev = map_cmplx_pos
        map_cmplx_pos = {}
        for pr_key, u_elems in map_prev.items():
            store = set(range(len(u_elems)))
            if len(u_elems) > 2:
                elems = sorted(u_elems)
                i = 1
                while i < len(elems):
                    if elems[i][0] == elems[i - 1][0]:
                        sum_costsA = numpy.sum(
                            [cmplx_candidates[cci].getCost() for cci in elems[i][1]])
                        sum_costsB = numpy.sum(
                            [cmplx_candidates[cci].getCost() for cci in elems[i - 1][1]])
                        if sum_costsA < sum_costsB:
                            elems.pop(i - 1)
                        else:
                            elems.pop(i)
                    else:
                        i += 1
                cpkids = [e[1] for e in elems]
                # FIX (  numpy.array  )  to not send a list to extract_cycles_fold()
                t0s = numpy.array([e[0] for e in elems])

                costSpare = cmplx_candidates[cpkids[0][0]].getCostNoE(
                    data_details)
                bound_dE = numpy.log2(data_details["deltaT"] + 1) - 2
                chains_x, triples_x = extract_cycles_fold(t0s, cmplx_candidates[cpkids[0][0]].getEventTuple(
                ), data_details, bound_dE, -costSpare, -costSpare)
                for triple in triples_x:
                    pr_key_kid = "%d+%s*%d" % (triple[1],
                                               pr_key, len(triple[-1]))
                    if pr_key_kid not in map_cmplx_pos:
                        map_cmplx_pos[pr_key_kid] = []
                    comb_kids = []
                    for cc in triple[-1]:
                        comb_kids.extend(cpkids[cc])
                    map_cmplx_pos[pr_key_kid].append(
                        (t0s[triple[-1][0]], comb_kids))
                    store.difference_update(triple[-1])

                for chain in chains_x:
                    pM = computePeriod(chain["occs"])
                    pr_key_kid = "%d+%s*%d" % (pM, pr_key, len(chain["occs"]))
                    if pr_key_kid not in map_cmplx_pos:
                        map_cmplx_pos[pr_key_kid] = []
                    comb_kids = []
                    for cc in chain["pos"]:
                        comb_kids.extend(cpkids[cc])
                    map_cmplx_pos[pr_key_kid].append(
                        (chain["occs"][0], comb_kids))
                    store.difference_update(chain["pos"])

            for epp in store:
                if not type(u_elems[epp][1]) is int:
                    # size, pr_key, t0, indices in cmplx_candidates
                    keep.append(
                        (eval(pr_key.split("_")[-1]), pr_key, u_elems[epp][0], u_elems[epp][1]))

    keep.sort()
    nested_patts = []
    covered = set()
    prev_size = -1
    while len(keep) > 0:
        next_cp = keep.pop()
        if len(set(next_cp[-1]).difference(covered)) > 2 or next_cp[0] == prev_size:
            prev_size = next_cp[0]
            new_cand = prepare_candidate_nested(next_cp, cmplx_candidates)
            new_cand.computeCost(data_details)

            covered.update(next_cp[-1])
            nested_patts.append(new_cand)
    return nested_patts, covered


def getPidsSlice(patterns_props, pids, slice_size, col, max_v):
    """
    FIXME : to be explained

    Parameters
    ----------
    patterns_props
    pids
    slice_size
    col
    max_v

    Returns
    -------

    """
    if patterns_props[pids[-1], col] <= max_v:
        return pids
    elif patterns_props[pids[0], col] > max_v:
        return []
    ii = numpy.where(
        patterns_props[pids[::slice_size] + [pids[-1]], col] > max_v)[0]
    last_id = ((ii[0] - 1) * slice_size) + numpy.where(
        patterns_props[pids[(ii[0] - 1) * slice_size:ii[0] * slice_size + 1], col] > max_v)[0][0]
    return pids[:last_id]


def run_combine_horizontal(cpool, data_details, dcosts, nkey="V"):
    """
    Combine candidates horizontally


    Parameters
    ----------
    cpool
    data_details
    dcosts
    nkey

    Returns
    -------

    """
    if cpool.nbNewCandidates(nkey) == 0:
        return []

    pids = list(cpool.getSortedPids())
    patterns_props = cpool.getPropMat()

    pids_new = None
    Inew = patterns_props[pids, prop_map["new"]] == cpool.getNewKNum(nkey)
    if numpy.sum(Inew) == 0:
        return []
    if numpy.sum(Inew) < 500:
        pids_new = [pids[p] for p in numpy.where(Inew)[0]]

    keep_cands = {}
    drop_overlap = 0
    # for each pattern Pa in turn
    while len(pids) > 1:
        if pids_new is not None:
            if len(pids_new) > 0:
                j = 0
                while pids[j] != pids_new[0] and ((patterns_props[pids[j], prop_map["t0i"]] + patterns_props[
                    pids[j], prop_map["p0"]]) < patterns_props[pids_new[0], prop_map["t0i"]]):
                    j += 1
                if pids[j] == pids_new[0]:
                    pids_new.pop(0)
                    if len(pids_new) == 0:  # last new pids reached -> last round
                        pids = [pids[j]]
                        j = 0
                pids = pids[j:]

        i = pids.pop(0)
        if len(pids) == 0:
            continue

        # find other patterns Pb such that:
        # (1) don"t come from the same candidate
        # (2) first occurrence of Pb appears before the second repetition of Pa
        # (3) pb-pa <= 2 (cum_E of Pb) / r(r-1) with r = min(ra, rb)

        # (2)
        next_it = patterns_props[i, prop_map["t0i"]] + patterns_props[i, prop_map["p0"]]
        i_new = patterns_props[i, prop_map["new"]] == cpool.getNewKNum(nkey)
        cmp_ids = []
        if i_new:  # i is new, compare to both new and old
            cmp_ids = numpy.array(getPidsSlice(
                patterns_props, pids, 500, prop_map["t0i"], next_it))
        else:  # i is old, only compare to new
            if pids_new is not None:
                ppp = numpy.array(pids_new)
                cmp_ids = ppp[patterns_props[ppp,
                prop_map["t0i"]] <= next_it]
            else:
                ppp = numpy.array(getPidsSlice(
                    patterns_props, pids, 500, prop_map["t0i"], next_it))
                if len(ppp) > 0:
                    cmp_ids = ppp[patterns_props[ppp, prop_map["new"]] == cpool.getNewKNum(nkey)]
        ###

        sel_ids = []
        if len(cmp_ids) > 0:
            # (1)
            sel_ids = cmp_ids[patterns_props[cmp_ids, prop_map["cid"]]
                              != patterns_props[i, prop_map["cid"]]]

        if len(sel_ids) > 0:
            # (3)
            rmins = 1. * numpy.minimum(
                patterns_props[i, prop_map["r0"]] - patterns_props[i, prop_map["offset"]],
                patterns_props[sel_ids, prop_map["r0"]] - patterns_props[
                    sel_ids, prop_map["offset"]])
            sel_ids = sel_ids[numpy.abs(patterns_props[sel_ids, prop_map["p0"]] - patterns_props[i,
            prop_map["p0"]]) <= 2. * patterns_props[sel_ids, prop_map["cumEi"]] / (
                                      rmins * (rmins - 1))]

        for j in sel_ids:
            cand_pids = (i, j)
            cand_cids = tuple(
                [patterns_props[cci, prop_map["cid"]] for cci in cand_pids])
            cands = [cpool.getCandidate(cci) for cci in cand_cids]
            if (cands[0].getEvent() == cands[1].getEvent()) and (
                    patterns_props[i, prop_map["t0i"]] == patterns_props[j, prop_map["t0i"]]):
                continue
            if len(set(cands[0].getEvOccs()).intersection(cands[1].getEvOccs())) > 0:
                drop_overlap += 1
                continue

            r0 = numpy.min(patterns_props[cand_pids, prop_map["r0"]] -
                           patterns_props[cand_pids, prop_map["offset"]])
            p0 = patterns_props[cand_pids[0], prop_map["p0"]]
            new_cand = prepare_candidate_concats(
                cands, p0, r0, patterns_props[cand_pids, prop_map["offset"]])
            new_cand.computeCost(data_details)
            if CHECK_HORDER and patterns_props[i, prop_map["t0i"]] == patterns_props[
                j, prop_map["t0i"]] and \
                    (numpy.abs(patterns_props[cand_pids[0], prop_map["p0"]] - patterns_props[
                        cand_pids[1], prop_map["p0"]]) <= 2. * patterns_props[
                         cand_pids[1], prop_map["cumEi"]] / (
                             r0 * (r0 - 1))):  # Equivalent flipped (same starting point)

                new_candX = prepare_candidate_concats([cands[1], cands[0]],
                                                      patterns_props[cand_pids[1], prop_map["p0"]], r0,
                                                      patterns_props[[
                                                          cand_pids[1], cand_pids[0]], prop_map["offset"]])
                new_candX.computeCost(data_details)

                if (new_candX.getCost() < new_cand.getCost()) or (
                        new_cand.getCost() == new_candX.getCost() and new_candX.getEventTuple() <
                        new_cand.getEventTuple()):
                    new_cand = new_candX
                    cand_pids = (cand_pids[1], cand_pids[0])

            sum_cost = numpy.sum([c.getCost() for c in cands])
            sum_nboccs = numpy.sum([c.getNbOccs() for c in cands])

            if new_cand.getCostRatio() < (sum_cost / sum_nboccs):
                cov = set().union(*[c.getEvOccs() for c in cands])
                residuals = cov.difference(new_cand.getEvOccs())
                cresiduals = numpy.sum([dcosts[o[1]] for o in residuals])

                if (new_cand.getCost() + cresiduals) / (new_cand.getNbOccs() + len(residuals)) < (
                        sum_cost / sum_nboccs):
                    keep_cands[cand_pids] = new_cand

                    for pp in numpy.where(patterns_props[i, prop_map["cid"]] == patterns_props[
                        pids, prop_map["cid"]])[0][::-1]:
                        pids.pop(pp)
                    if pids_new is not None:
                        for pp in numpy.where(patterns_props[i, prop_map["cid"]] == patterns_props[
                            pids_new, prop_map["cid"]])[0][::-1]:
                            pids_new.pop(pp)
                        if len(pids_new) == 0:
                            pids = []

    selected_ids = filter_candidates_topKeach(keep_cands, k=TOP_KEACH)
    keep_cands = dict([(k, keep_cands[k]) for k in selected_ids])
    graph_candidates = {}
    for cand_pids in selected_ids:
        for cci in [0, 1]:
            if cand_pids[cci] not in graph_candidates:
                graph_candidates[cand_pids[cci]] = set([cand_pids[1 - cci]])
            else:
                graph_candidates[cand_pids[cci]].add(cand_pids[1 - cci])

    collect = []
    bronKerbosch3Plus(graph_candidates, collect, set(graph_candidates.keys()))
    for cand_pids_unsrt in collect:
        cand_pids = sorted(cand_pids_unsrt,
                           key=lambda x: (patterns_props[x, prop_map["t0i"]],
                                          tuple(int(x) for x in cpool.getCandidate(
                                              patterns_props[x, prop_map["cid"]]).getEventTuple())))
        new_cand = makeCandOnOrder(
            cand_pids, data_details, patterns_props, cpool)

        if CHECK_HORDER and len(set(patterns_props[cand_pids, prop_map["t0i"]])) < len(cand_pids):
            ppids = [s for s in selected_ids if (
                    s[0] in cand_pids_unsrt) and (s[1] in cand_pids_unsrt)]
            ord_c = dict([(s, 0) for s in cand_pids])
            for ppid in ppids:
                ord_c[ppid[0]] += 1
            cand_pidsX = sorted(ord_c.keys(), key=lambda x: -ord_c[x])
            if cand_pidsX != cand_pids:
                new_candX = makeCandOnOrder(
                    cand_pidsX, data_details, patterns_props, cpool)

                if new_candX.getCost() < new_cand.getCost():
                    new_cand = new_candX
                    cand_pids = cand_pidsX

        if CHECK_HORDER:
            cand_pidsY = sortPids(patterns_props, cand_pids_unsrt)
            if cand_pidsY != cand_pids:
                new_candY = makeCandOnOrder(
                    cand_pidsY, data_details, patterns_props, cpool)
                if new_candY.getCost() < new_cand.getCost():
                    new_cand = new_candY
                    cand_pids = cand_pidsY

        sum_cost = numpy.sum([c.getCost() for c in cands])
        sum_nboccs = numpy.sum([c.getNbOccs() for c in cands])

        if new_cand.getCostRatio() < (sum_cost / sum_nboccs):
            keep_cands[tuple(cand_pids)] = new_cand

    selected = list(keep_cands.values())
    substitute_factorized(selected, data_details)
    return selected


def makeCandOnOrder(cand_pids, data_details, patterns_props, cpool):
    """
    FIXME : to be explained

    Parameters
    ----------
    cand_pids
    data_details
    patterns_props
    cpool

    Returns
    -------

    """
    cands = [cpool.getCandidate(
        patterns_props[cci, prop_map["cid"]]) for cci in cand_pids]
    r0 = numpy.min(patterns_props[cand_pids, prop_map["r0"]] -
                   patterns_props[cand_pids, prop_map["offset"]])
    p0 = patterns_props[cand_pids[0], prop_map["p0"]]
    new_cand = prepare_candidate_concats(
        cands, p0, r0, patterns_props[cand_pids, prop_map["offset"]])
    new_cand.computeCost(data_details)
    return new_cand


def filter_candidates_cover(cands, dcosts, min_cov=1, adjust_occs=False, cis=None):
    """
    Filters a list of candidates based on their coverage and cost efficiency.

    Parameters:
    cands (dict or list): A dictionary or list of Candidate objects.
    dcosts (dict): A dictionary of costs for each data point.
    min_cov (int): The minimum number of data points a candidate must cover to be selected. Default is 1.
    adjust_occs (bool): Whether to adjust occurrences of patterns. Default is False.
    cis (list): A list of candidate indices to consider. If not provided, all candidates will be considered.

    Returns:
    list: A list of selected candidate indices.

    Example
    ------
    >>> cands = [Candidate(...), Candidate(...), ...]
    >>> dcosts = {'data1': 0.5, 'data2': 0.8, ...}
    >>> selected = filter_candidates_cover(cands, dcosts, min_cov=2, adjust_occs=True, cis=[0, 2, 5])
    """
    if cis is None:
        if type(cands) is dict:
            cis = list(cands.keys())
        else:
            cis = list(range(len(cands)))
    for ci in cis:
        cands[ci].initUncovered()

    selected = []
    covered = set()
    if len(dcosts) == 0:
        return selected
    cis.sort(key=lambda x: cands[x].getCostUncoveredRatio())
    max_eff = numpy.min(list(dcosts.values()))

    while len(cis) > 0:
        nxti = cis.pop(0)
        if cands[nxti].getCostUncoveredRatio() <= max_eff:
            if cands[nxti].getNbUncovered() >= min_cov and cands[nxti].isEfficient(dcosts):
                if not cands[nxti].isPattern() and adjust_occs:
                    cands[nxti].adjustOccs()

                selected.append(nxti)
                covered.update(cands[nxti].getUncovered())

                if cands[nxti].getNbUncovered() > 0:
                    i = 0
                    while i < len(cis):
                        if cands[cis[i]].updateUncovered(cands[nxti].getUncovered()) < min_cov:
                            cis.pop(i)
                        elif (max_eff >= 0) and (cands[cis[i]].getCostUncoveredRatio() > max_eff):
                            cis.pop(i)
                        else:
                            i += 1
                    cis.sort(key=lambda x: cands[x].getCostUncoveredRatio())

    return selected


def filter_candidates_topKeach(cands, k=2):
    """
    FIXME : to be explained

    Parameters
    ----------
    cands
    k

    Returns
    -------

    """
    counts_cover = {}

    if type(cands) is dict:
        cis = list(cands.keys())
    else:
        cis = list(range(len(cands)))
    if len(cis) <= k:
        return cis

    cis.sort(key=lambda x: (cands[x].getCostRatio(), x))
    selected = []
    while len(cis) > 0:
        nxti = cis.pop(0)

        if cands[nxti].satisfiesMaxCountCover(counts_cover, k):
            cands[nxti].updateCountCover(counts_cover)
            selected.append(nxti)
    return selected


def substitute_factorized(cands, data_details):
    """
    FIXME : to be explained

    Parameters
    ----------
    cands
    data_details

    Returns
    -------

    """
    for i in range(len(cands)):
        ext = cands[i].factorizePattern()
        if len(ext) > 0:
            ii = numpy.argmin([cands[i].getCost()] +
                              [e.computeCost(data_details) for e in ext])
            if ii > 0:
                cands[i] = ext[ii - 1]


def mine_seqs(seqs, complex=True, max_p=None):
    """
    Mines cycles and patterns from a set of sequences.

    Parameters
    ----------
    seqs : DataSequence or list of str
        A DataSequence object or a list of sequences to mine patterns from.
    complex : bool, optional
        Specifies if the sequences are complex (True) or simple (False). Default is True.
    max_p : int or None, optional
        The maximum size of the patterns to mine. Default is None.

    Returns
    -------
    dict
        A dictionary containing the mined patterns and statistics.
    """
    MINE_CPLX = True if complex else False

    if type(seqs) is DataSequence:
        ds = seqs
    else:
        ds = DataSequence(seqs)

    results = {}

    data_details = ds.getDetails()
    dcosts = dict([(alpha, cost_one(data_details, alpha))
                   for alpha in data_details["nbOccs"].keys() if alpha != -1])

    tic = datetime.datetime.now()
    dT_sel = datetime.timedelta()

    results["TIME Start"] = str(tic)

    cpool = CandidatePool()
    evs = ds.getEvents()  # list containing all events names

    results["ev"] = []
    results["alpha"] = []
    results["len(seq)"] = []
    results["TIME run"] = []
    for alpha, ev in enumerate(evs):  # alpha is the mapping number associated to the event `ev`
        tic_ev = datetime.datetime.now()
        seq = ds.getSequence(alpha)  # return the associated timestamps associated to the event number alpha

        results["ev"].append(ev)
        results["alpha"].append(alpha)
        results["len(seq)"].append(len(seq))

        cycles_alpha = mine_cycles_alpha(
            seq, alpha, data_details, dcosts[alpha], max_p=max_p)

        cpool.addCands(cycles_alpha, costOne=dcosts[alpha])
        tac_ev = datetime.datetime.now()
        results["TIME run"].append(str(tac_ev - tic_ev))

    tac_init = datetime.datetime.now()
    results["[TIME] simple cycle"] = str(tac_init - tic)

    tic_sel = datetime.datetime.now()
    cdict = cpool.getCandidates()

    simple_cids = list(cdict.keys())
    results["Nb candidates"] = len(cdict)
    # print("\n Nb candidates",  len(cdict), "\n")
    # print("\n simple_cids", simple_cids, "\n")
    # print("\n\n\n cdict", simple_cids, "\n\n\n")
    results["time candidates"] = str(tic_sel)


    tac_sel = datetime.datetime.now()
    dT_sel += (tac_sel - tic_sel)
    results["[INTER] Simple selection DT"] = str(tac_sel - tic_sel)
    results["[INTER] Simple selection TIME"] = str(tac_sel)
    #############

    nkeyV, nkeyH = (None, None)
    roundi = 0

    results["nkeyV"] = []
    results["nkeyH"] = []
    results["[TIME] Combination round"] = []
    results["[DT] Combination round"] = []
    results["[DT] V"] = []
    results["[DT] H"] = []
    results["[DT] C"] = []
    results["Simple selection"] = []

    while MINE_CPLX and cpool.nbNewCandidates() > 0:
        roundi += 1

        tic_round = datetime.datetime.now()
        candsV = run_combine_vertical(
            cpool, data_details, nkeyH)
        tac_rV = datetime.datetime.now()
        candsH = run_combine_horizontal(
            cpool, data_details, dcosts, nkeyV)
        tac_rH = datetime.datetime.now()

        nkeyV, nkeyH = ("V%d" % roundi, "H%d" % roundi)

        results["nkeyV"].append(nkeyV)
        results["nkeyH"].append(nkeyH)

        cpool.resetNew()
        cpool.addCands(candsV, nkeyV)
        cpool.addCands(candsH, nkeyH)
        tac_round = datetime.datetime.now()

        results["[TIME] Combination round"].append(roundi)
        results["[DT] Combination round"].append(str(tac_round - tic_round))
        results["[DT] V"].append(str(tac_rV - tic_round))
        results["[DT] H"].append(str(tac_rH - tac_rV))
        results["[DT] C"].append(str(tac_round - tac_rH))

        simple_selection = {}
        simple_selection_side = []
        simple_selection_nb_candidates = []
        simple_selection_nb_time = []
        simple_selection_finalside = []
        simple_selection_finalDT = []
        simple_selection_finalTime = []

        if roundi == 1:
            for (side, nks) in [("V", [nkeyV]), ("H", [nkeyH]), ("V+H", [nkeyV, nkeyH])]:
                tic_sel = datetime.datetime.now()
                cdict = cpool.getCandidates()

                # print("\n\n ********** COMPOSED Candidates**********")
                # print(" cdict", cdict)

                to_filter = list(simple_cids)
                for nk in nks:
                    to_filter.extend(cpool.getNewCids(nk))

                simple_selection_side.append(side)
                simple_selection_nb_candidates.append(len(to_filter))
                simple_selection_nb_time.append(str(tic_sel))

                tac_sel = datetime.datetime.now()
                dT_sel += (tac_sel - tic_sel)

                simple_selection_finalside.append(side)
                simple_selection_finalDT.append(str(tac_sel - tic_sel))
                simple_selection_finalTime.append(str(tac_sel))

        simple_selection["side"] = simple_selection_side
        simple_selection["nb_candidates"] = simple_selection_nb_candidates
        simple_selection["nb_time"] = simple_selection_nb_time
        simple_selection["finalside"] = simple_selection_finalside
        simple_selection["finalDT"] = simple_selection_finalDT
        simple_selection["finalTime"] = simple_selection_finalTime

        results["Simple selection"].append(simple_selection)

    tac_comb = datetime.datetime.now()

    results["[TIME] Combinations"] = str(tac_comb - tac_init)

    cdict = cpool.getCandidates()

    results["Final_selection_nb_candidates"] = len(cdict)
    results["Final_selection_TIME"] = str(tac_comb)

    selected = filter_candidates_cover(
        cdict, dcosts, min_cov=3, adjust_occs=True)
    pc = PatternCollection([cdict[c].getPattT0E() for c in selected])
    tac = datetime.datetime.now()

    results["Final DT selection"] = str(tac - tac_comb)
    results["Final TIME selection"] = str(tac)

    results["Final Mining DT"] = str(tac - tac_comb)
    results["Final Mining inter DT"] = str((tac - tic) - dT_sel)
    results["Final DT"] = str((tac - tic) - dT_sel)

    return cpool, ds, pc
