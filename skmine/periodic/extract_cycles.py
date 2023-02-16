import datetime
import pdb

import numpy

from .class_patterns import DataSequence, computeLengthRC, computeLengthCycle, computeLengthResidual, computePeriod, \
    cost_triple, cost_one

DYN_BLOCK_SIZE = 100
STEP_SIZE = 80


# make a "original" solution from synthetic data generation details


def make_solution(dets, alpha):
    solution = []
    uncovered = set()
    for ci, d in dets.items():
        if ci == -1 or len(d[0]) < 3:
            uncovered.update(d[0])
        else:
            solution.append(
                {"alpha": alpha, "occs": d[0], "p": d[-1]["period"]})

    solution.sort(key=lambda x: x["occs"][0])
    if len(uncovered) > 0:
        solution.append({"alpha": alpha, "occs": sorted(uncovered)})
    return solution


# DYNAMIC PROGRAMMING WITH OVERLAPS
# extract segments boundaries from dynamic programming output
def recover_splits_rec_ov(spoints, ia, iz, depth=0):
    if depth > 200:
        print("Exceeded depth ", depth)
        # pdb.set_trace()
        return []
    if (ia, iz) in spoints:
        if spoints[(ia, iz)] is None:
            return [(ia, iz)]
        else:
            im = spoints[(ia, iz)]
            if im > ia:
                return recover_splits_rec_ov(spoints, ia, im, depth + 1) + recover_splits_rec_ov(spoints, im, iz,
                                                                                                 depth + 1)
    return [(i, i) for i in range(ia, iz + 1)]


# fill in dynamic programming table


def compute_table_dyn_ov(occs, alpha, data_details):
    ilast = len(occs)
    score_res = computeLengthResidual(
        data_details, {"alpha": alpha, "occs": [0]})
    scores = {}
    spoints = {}

    for ia in range(ilast - 2):
        iz = ia + 2
        score_best = computeLengthCycle(data_details, {"p": None, "alpha": alpha, "occs": [
            occs[i] for i in range(ia, iz + 1)]})
        # print("+ Test-3\t", ia, ia, iz, "\t", score_best, 3*score_res )
        if score_best < 3 * score_res:
            scores[(ia, iz)] = score_best
            spoints[(ia, iz)] = None
        else:
            scores[(ia, iz)] = 3 * score_res
            spoints[(ia, iz)] = ia

    for k in range(3, ilast):
        for ia in range(ilast - k):
            iz = ia + k
            score_best = computeLengthCycle(data_details, {"p": None, "alpha": alpha, "occs": [
                occs[i] for i in range(ia, iz + 1)]})
            spoint_best = None
            # print("+ Test\t", ia, ia, iz, "\t", score_best)

            for im in range(ia + 1, iz):
                # print("- Test", ia, im, iz)
                if im - ia <= 2:
                    score_left = score_res * (im - ia + 1)
                else:
                    score_left = scores[(ia, im)]
                if iz - im <= 2:
                    score_right = score_res * (iz - im + 1)
                else:
                    score_right = scores[(im, iz)]

                if score_left + score_right < score_best:
                    score_best = score_left + score_right
                    spoint_best = im

            scores[(ia, iz)] = score_best
            spoints[(ia, iz)] = spoint_best
    return scores, spoints


# DYNAMIC PROGRAMMING
# extract segments boundaries from dynamic programming output
def recover_splits_rec(spoints, ia, iz, depth=0, singletons=True):
    # if depth > 200:
    #     print("Exceeded depth ", depth)
    #     # pdb.set_trace()
    #     return []
    if (ia, iz) in spoints:
        if spoints[(ia, iz)] is None:
            return [(ia, iz)]
        else:
            im = spoints[(ia, iz)]
            if im >= 0:
                return recover_splits_rec(spoints, ia, im, depth + 1, singletons) + recover_splits_rec(spoints, im + 1,
                                                                                                       iz, depth + 1,
                                                                                                       singletons)
    if singletons:
        return [(i, i) for i in range(ia, iz + 1)]
    else:
        return []


# fill in dynamic programming table


def compute_table_dyn(occs, alpha, data_details):
    ilast = len(occs)
    score_res = computeLengthResidual(
        data_details, {"alpha": alpha, "occs": [0]})
    scores = {}
    spoints = {}

    for ia in range(ilast - 2):
        iz = ia + 2
        score_best = computeLengthCycle(data_details, {"p": None, "alpha": alpha, "occs": [
            occs[i] for i in range(ia, iz + 1)]})
        # print("+ Test-3\t", ia, ia, iz, "\t", score_best, 3*score_res )
        if score_best < 3 * score_res:
            scores[(ia, iz)] = score_best
            spoints[(ia, iz)] = None
        else:
            scores[(ia, iz)] = 3 * score_res
            spoints[(ia, iz)] = -1
    deb = False

    for k in range(3, ilast):
        for ia in range(ilast - k):
            iz = ia + k

            score_best = computeLengthCycle(data_details, {"p": None, "alpha": alpha, "occs": [
                occs[i] for i in range(ia, iz + 1)]})
            spoint_best = None
            # print("+ Test\t", ia, ia, iz, "\t", score_best)

            for im in range(ia, iz):
                if im - ia + 1 <= 2:
                    score_left = score_res * (im - ia + 1)
                else:
                    score_left = scores[(ia, im)]
                if iz - im <= 2:
                    score_right = score_res * (iz - im)
                else:
                    score_right = scores[(im + 1, iz)]

                if score_left + score_right < score_best:
                    score_best = score_left + score_right
                    spoint_best = im

            scores[(ia, iz)] = score_best
            spoints[(ia, iz)] = spoint_best
    return scores, spoints


def combine_splits(splits, adj_splits):
    if len(adj_splits) > 0:
        if len(splits) == 0:
            splits = adj_splits
        else:
            # append and combine splits
            prev_i = 0
            popped = 0
            ready = False

            while not ready:
                while prev_i < len(splits) and splits[prev_i][1] < adj_splits[0][0]:
                    prev_i += 1

                if prev_i >= len(splits):
                    # reached the end of splits
                    ready = True
                elif adj_splits[0][0] <= splits[prev_i][0]:
                    # first adj starts before or at current splits
                    ready = True
                else:
                    # first adj starts after current splits
                    if adj_splits[0][1] <= splits[prev_i][1]:
                        # first adj ends before current splits
                        adj_splits.pop(0)
                        popped += 1
                        if len(adj_splits) == 0:
                            ready = True
                    else:
                        adj_splits[0] = (splits[prev_i][0], adj_splits[0][1])
                        ready = True

            splits = splits[:prev_i] + adj_splits
    return splits


def compute_cycles_dyn(occs, alpha, data_details, residuals=True):
    ilast = len(occs)
    if DYN_BLOCK_SIZE == 0 or ilast > 2 * DYN_BLOCK_SIZE:
        # compute best split points on ovelapping blocks of the sequence, for efficiency
        splits = []
        pstart = 0
        costs = {}
        while pstart + .5 * DYN_BLOCK_SIZE < ilast:
            next_block = occs[pstart:pstart + DYN_BLOCK_SIZE]
            blast = len(next_block)

            scores, spoints = compute_table_dyn(
                next_block, alpha, data_details)
            bsplits = recover_splits_rec(spoints, 0, blast - 1, singletons=False)
            adj_splits = [(ia + pstart, iz + pstart) for (ia, iz) in bsplits]

            costs.update(
                dict([((ia + pstart, iz + pstart), scores[(ia, iz)]) for (ia, iz) in bsplits]))
            splits = combine_splits(splits, adj_splits)

            pstart += STEP_SIZE
    else:
        scores, spoints = compute_table_dyn(occs, alpha, data_details)
        splits = recover_splits_rec(spoints, 0, ilast - 1, singletons=False)
        costs = scores

    cycles = []
    covered = set()
    for si, s in enumerate(splits):
        if s[1] - s[0] > 1:  # contains at least 3 elements
            cov = [occs[i] for i in range(s[0], s[1] + 1)]
            prd = computePeriod(cov)
            if s in costs:
                cst = costs[s]
            else:
                cst = computeLengthCycle(
                    data_details, {"p": prd, "alpha": alpha, "occs": cov})
            cycles.append({"alpha": alpha, "occs": cov, "p": prd, "cost": cst})
            covered.update(cov)

    cycles.sort(key=lambda x: x["occs"][0])

    if residuals:
        uncovered = set(occs) - covered
        if len(uncovered) > 0:
            cycles.append({"alpha": alpha, "occs": sorted(uncovered)})
    return cycles


# FOLDING COMPUTATION
def select_chains(chains, cycles, covered, occs, alpha, cost3):
    # print("selecting chains...")
    chains.sort(key=lambda x: (x["cost"] / len(x["uncov"])))
    ###############################################
    # pdb.set_trace()
    while len(chains) > 0:
        nxt = chains.pop(0)
        if len(nxt["uncov"]) >= 3:
            okk = nxt["occs"]
            if len(nxt["uncov"]) < len(nxt["pos"]):
                mni, mxi = (0, len(nxt["pos"]) - 1)
                while nxt["pos"][mni] not in nxt["uncov"]:
                    mni += 1
                while nxt["pos"][mxi] not in nxt["uncov"]:
                    mxi -= 1
                okk = [occs[nxt["pos"][kk]] for kk in range(mni, mxi + 1)]
            if nxt["cost"] / len(nxt["uncov"]) < cost3 / 3.:
                cycles.append({"alpha": alpha, "occs": okk, "p": None})
                covered.update(nxt["uncov"])

                # print(":::", nxt["cost"], cost3, len(nxt["uncov"]), len(okk), nxt["cost"]/len(nxt["uncov"]),
                # nxt["cost"]/len(nxt["uncov"]) < cost3/3.)
                # computeLengthCycle(nbOccs, deltaT, {"p": None, "alpha": alpha, "occs": okk})
                # computeLengthResidual(nbOccs, deltaT, {"alpha": alpha, "occs": okk})

        if len(nxt["uncov"]) > 0:
            i = 0
            while i < len(chains):
                chains[i]["uncov"].difference_update(nxt["uncov"])
                if len(chains[i]["uncov"]) == 0:
                    chains.pop(i)
                else:
                    i += 1
            chains.sort(key=lambda x: (x["cost"] / len(x["uncov"])))
    return cycles
    # print("--- t:", datetime.datetime.now()-tic)
    # tic = datetime.datetime.now()


def select_triples(triples, cycles, covered, occs, alpha):
    # print("selecting triples...")
    triples.sort()
    for t in triples:
        if len(covered.intersection(t[-1])) == 0:
            covered.update(t[-1])
            cycles.append({"alpha": alpha, "occs": [
                occs[tt] for tt in t[-1]], "p": None})
    # print("--- t:", datetime.datetime.now()-tic)
    return cycles


def extract_cycles_fold(occs, alpha, data_details, bound_dE, eff_trip, eff_chain, max_p=None):
    if len(occs) < 2000:
        return extract_cycles_fold_sub(occs, alpha, data_details, bound_dE, eff_trip, eff_chain, max_p=max_p)
    else:
        chains, triples = extract_cycles_fold_sub(
            occs[:1500], alpha, data_details, bound_dE, eff_trip, eff_chain, max_p=max_p)
        for i in range(1, int(len(occs) / 1500)):
            chains_tmp, triples_tmp = extract_cycles_fold_sub(
                occs[i * 1500:(i + 1) * 1500], alpha, data_details, bound_dE, eff_trip, eff_chain, i * 1500,
                max_p=max_p)
            chains.extend(chains_tmp)
            triples.extend(triples_tmp)
        return chains, triples


def extract_cycles_fold_sub(occs, alpha, data_details, bound_dE, eff_trip, eff_chain, offset=0, max_p=None):
    # centers = sorted(range(len(occs)), key=lambda x: numpy.abs(x-len(occs)/2), reverse=True)
    if len(occs) > len(set(occs)):
        pdb.set_trace()
    centers = list(range(len(occs) - 1, 0, -1))
    pairs_chain_test = {}
    pairs_chain_fwd = {}
    pairs_chain_bck = {}
    triples_tmp = {}

    # tic = datetime.datetime.now()
    while len(centers) > 0:
        i = centers.pop()
        # dps_block = sorted([dp for (dp, ctrs) in centers_dps_block.items() if i in ctrs])
        # print("-- %d DPS BLOCK: %s" % (i, dps_block))

        # compute distance to occurrence before and after center position

        # occs = pd.Int64Index(occs) # FIX when occs became a list

        bef = occs[:i][::-1] - occs[i]
        aft = occs[i + 1:] - occs[i]
        if max_p is not None:
            aft = aft[aft < max_p]
            bef = bef[bef > -max_p]
        seen_ps = []
        ia, ib = (0, 0)
        while ia < len(aft) and ib < len(bef):
            # while ia < len(aft)-1 and ib < len(bef)-1:
            if ib < len(bef) - 1 and numpy.abs(aft[ia] + bef[ib]) > numpy.abs(aft[ia] + bef[ib + 1]):
                # print("\tskip b", (i-ib-1, i, i+ia+1), (numpy.abs(aft[ia]+bef[ib]), numpy.abs(aft[ia]+bef[ib+1])))
                ib += 1
            elif ia < len(aft) - 1 and numpy.abs(aft[ia] + bef[ib]) > numpy.abs(aft[ia + 1] + bef[ib]):
                # print("\tskip a", (i-ib-1, i, i+ia+1), (numpy.abs(aft[ia]+bef[ib]), numpy.abs(aft[ia+1]+bef[ib])))
                ia += 1
            else:
                deltaE = numpy.abs(aft[ia] + bef[ib])
                # if deltaE >= cost_trip:
                #     print("\tskip D", (i-ib-1, i, i+ia+1), deltaE, (numpy.min([aft[ia], -bef[ib]]), numpy.max([aft[
                #     ia], -bef[ib]])))

                if deltaE < bound_dE:
                    # print("store", (i-ib-1, i, i+ia+1), deltaE, (numpy.min([aft[ia], -bef[ib]]), numpy.max([aft[
                    # ia], -bef[ib]])))
                    dp = numpy.min([aft[ia], -bef[ib]])
                    dp_alt = numpy.max([aft[ia], -bef[ib]])
                    ipp = 0
                    while 0 <= ipp < len(seen_ps):
                        # if (dp % seen_ps[ipp] == 0 and dp / seen_ps[ipp] < 5) or (dp_alt % seen_ps[ipp] == 0  and
                        # dp_alt / seen_ps[ipp] < 5):
                        # if (dp % seen_ps[ipp] == 0):
                        if dp == 2 * seen_ps[ipp]:
                            seen_ps.append(dp)
                            ipp = -seen_ps[ipp]
                        # elif (dp_alt % seen_ps[ipp] == 0):
                        elif dp_alt == 2 * seen_ps[ipp]:
                            seen_ps.append(dp_alt)
                            ipp = -seen_ps[ipp]
                        else:
                            ipp += 1
                    if ipp >= 0:
                        costC = cost_triple(data_details, alpha, dp, deltaE)
                        # cost_check = computeLengthCycle(data_details, {"p": dp, "alpha": alpha, "occs": [occs[xo]
                        # for xo in [i-ib-1, i, i+ia+1]]})
                        # if int(10000*cost_check) != int(10000*costC):
                        #     pdb.set_trace()
                        # if costC <= bound_dE: ### also apply on cost of triple

                        seen_ps.append(dp)
                        if (i - ib - 1, i) not in pairs_chain_fwd:
                            if (i - ib - 1, i) in pairs_chain_test:
                                pairs_chain_fwd[(i - ib - 1, i)] = (deltaE, dp, i + ia + 1)
                                pairs_chain_bck[(i - ib - 1, i)] = pairs_chain_test.pop((i - ib - 1, i))[:-1]

                        elif pairs_chain_fwd[(i - ib - 1, i)][0] > deltaE:
                            pairs_chain_fwd[(i - ib - 1, i)] = (deltaE, dp, i + ia + 1)

                        if (i, i + ia + 1) not in pairs_chain_test or pairs_chain_test[(i, i + ia + 1)][0] > deltaE:
                            pairs_chain_test[(i, i + ia + 1)] = (deltaE, dp, i - ib - 1, costC)

                ia += 1
                ib += 1
        if i % 100 == 0 and i > 0:
            # print(len(pairs_chain_test))
            # triples.extend([(v[0], v[1], (v[-1], k[0], k[1])) for (k,v) in pairs_chain_test.items() if k[1] < i])
            # triples_tmp: (i, ia) -> (deltaE, prd, ib, cost)
            triples_tmp.update(
                dict([(k, v) for (k, v) in pairs_chain_test.items() if k[1] < i]))
            pairs_chain_test = dict(
                [(k, v) for (k, v) in pairs_chain_test.items() if k[1] >= i])
            # print(len(pairs_chain_test))

    # triples.extend([(v[0], v[1], (v[-1], k[0], k[1])) for (k,v) in pairs_chain_test.items()])
    triples_tmp.update(pairs_chain_test)
    # print("--- t:", datetime.datetime.now()-tic)
    # tic = datetime.datetime.now()
    # print("chaining...")
    chains = []
    # if len(pairs_chain_fwd) > 100:
    #     print("Nb chains fwd %d, hist bin dE %s" % (len(pairs_chain_fwd), numpy.bincount([v[0] for v in
    #     pairs_chain_fwd.values()])))
    while len(pairs_chain_fwd) > 0:
        kf = min(pairs_chain_fwd.keys())
        current = [pairs_chain_bck[kf][-1], kf[0], kf[1]]
        while (current[-2], current[-1]) in pairs_chain_fwd:
            nxt = pairs_chain_fwd.pop((current[-2], current[-1]))
            # print(current[-2:], nxt)
            if (current[-1], nxt[-1]) in triples_tmp:
                del triples_tmp[(current[-1], nxt[-1])]
            current.append(nxt[-1])
        # pdb.set_trace()
        # if len(chains) > 100:
        #     pdb.set_trace()
        #     print("Nb chains fwd left ", len(pairs_chain_fwd))
        #     print("Nb chains", len(chains))

        occs_chain = [occs[c] for c in current]
        prd = computePeriod(occs_chain)
        cost = computeLengthCycle(
            data_details, {"p": prd, "alpha": alpha, "occs": occs_chain})
        if (eff_chain < 0 and cost < (-eff_chain) * (len(occs_chain) - 1)) or (cost / len(occs_chain) < eff_chain):
            chains.append({"alpha": alpha, "p": prd, "cost": cost,
                           "pos": current, "occs": occs_chain, "uncov": set(current)})
    # print(numpy.bincount([len(cc["pos"]) for cc in chains]))
    # triples: (deltaE, prd, cost, (ib, i, ia))

    # if (occs[i-ib-1], occs[i], occs[i+ia+1]) == (18125, 508285, 901652):
    # if (v[-2]+offset, k[0]+offset, k[1]+offset) == (3167, 3302, 3440): pdb.set_trace()
    triples = [(v[0], v[1], v[-1], (v[-2] + offset, k[0] + offset, k[1] + offset)) for (k, v)
               in triples_tmp.items() if (eff_trip < 0 and v[-1] < -2 * eff_trip) or (v[-1] / 3. < eff_trip)]
    # print("--- t:", datetime.datetime.now()-tic)
    # tic = datetime.datetime.now()
    return chains, triples


def compute_cycles_fold(occs, alpha, data_details, residuals=True, inc_triples=True):
    costOne = cost_one(data_details, alpha)
    bound_dE = numpy.log2(data_details["deltaT"] + 1) - 2
    chains, triples = extract_cycles_fold(
        occs, alpha, data_details, bound_dE, costOne, costOne)
    cycles = []
    covered = set()
    select_chains(chains, cycles, covered, occs, alpha, 3 * costOne)
    if inc_triples:
        select_triples(triples, cycles, covered, occs, alpha)

    for c in cycles:
        c["xp"] = computePeriod(c["occs"])
    cycles.sort(key=lambda x: x["xp"])

    if residuals:
        uncovered = set(occs).difference([occs[c] for c in covered])
        if len(uncovered) > 0:
            cycles.append({"alpha": alpha, "occs": sorted(uncovered)})
    return cycles


def run_test(occs, alpha, data_details, dets=None):
    results = []
    noc_cycl = [{"alpha": alpha, "occs": occs}]
    noc_cost = computeLengthRC(data_details, noc_cycl)
    results.append({"meth": "no cycles", "CL": noc_cost,
                    "RT": 0, "cycles": noc_cycl})

    noc_cycl = [{"alpha": alpha, "occs": occs, "p": None}]
    noc_cost = computeLengthRC(data_details, noc_cycl)
    results.append({"meth": "one cycle", "CL": noc_cost,
                    "RT": 0, "cycles": noc_cycl})

    # if dets is not None:
    #     sol = make_solution(dets, alpha)
    #     # print("ORG:", sol)
    #     sol_cost = computeLengthRC(nbOccs, deltaT, sol)
    #     results.append({"meth": "org_sol", "CL": sol_cost, "RT": 0, "cycles": sol})

    tic = datetime.datetime.now()
    dyn_cycles = compute_cycles_dyn(occs, alpha, data_details)
    # print("DYN:", dyn_cycles)
    elsp_dyn = datetime.datetime.now() - tic
    dyn_cost = computeLengthRC(data_details, dyn_cycles)
    results.append({"meth": "segments", "CL": dyn_cost,
                    "RT": elsp_dyn.total_seconds(), "cycles": dyn_cycles})

    # tic = datetime.datetime.now()
    # fld_cycles = compute_cycles_fld(occs, alpha, deltaT, nbOccs)
    # # print("FLD-OLD:", fld_cycles)
    # elsp_fld = datetime.datetime.now()-tic
    # fld_cost = computeLengthRC(nbOccs, deltaT, fld_cycles)
    # results.append({"meth": "fld-old", "CL": fld_cost, "RT": elsp_fld.total_seconds(), "cycles": fld_cycles})

    tic = datetime.datetime.now()
    fold_cycles = compute_cycles_fold(occs, alpha, data_details)
    # print("FLD:", fld_cycles)
    elsp_fold = datetime.datetime.now() - tic
    fold_cost = computeLengthRC(data_details, fold_cycles)
    results.append({"meth": "folding", "CL": fold_cost,
                    "RT": elsp_fold.total_seconds(), "cycles": fold_cycles})

    return results


if __name__ == "__main__":
    # comb_params = [[{"t0": 0, "length_org": 40, "period_org": 10, "beta": .25, "max": 0.2},
    #                 {"proba_add": 0.}]]
    comb_params = [[{"t0": 0, "length_org": 50, "period_org": 10},
                    {"t0": 1.1, "length_org": 40, "period_org": 34},
                    {"proba_add": 0.}]]

    # comb_params = [[{"t0": 0, "length_org": 5, "period_org": 10},
    #                 {"t0": 1., "length_org": 4, "period_org": 34},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 4, "period_org": 34},
    #                 {"t0": 1., "length_org": 5, "period_org": 10},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 4, "period_org": 34},
    #                {"t0": 1., "length_org": 5, "period_org": 10},
    #                {"t0": 1., "length_org": 6, "period_org": 3},
    #                {"t0": 1., "length_org": 5, "period_org": 10},
    #                {"t0": 1., "length_org": 4, "period_org": 34},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 4, "period_org": 34},
    #                {"t0": 1., "length_org": 5, "period_org": 5},
    #                {"t0": 1., "length_org": 6, "period_org": 23},
    #                {"t0": 1., "length_org": 5, "period_org": 5},
    #                {"t0": 1., "length_org": 4, "period_org": 34},
    #                 {"proba_add": 0.}]]

    # comb_params = [[{"t0": 0, "length_org": 300, "period_org": 10, "beta": .25, "max": 0.2},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 1000, "period_org": 10, "beta": .25, "max": 0.2},
    #                 {"proba_add": 0.2}],
    #                [{"t0": 0, "length_org": 1000, "period_org": 10, "beta": .25, "max": 0.2},
    #                 {"proba_add": 0.3}],
    #                 [{"t0": 0, "length_org": 1000, "period_org": 10, "beta": .01, "max": 0.2},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 1000, "period_org": 10, "beta": .01, "max": 0.2},
    #                 {"proba_add": 0.2}],
    #                [{"t0": 0, "length_org": 1000, "period_org": 10, "beta": .01, "max": 0.2},
    #                 {"proba_add": 0.3}]]

    # comb_params = [[{"t0": 0, "length_org": 100, "period_org": 10},
    #                 {"t0": 0.4, "length_org": 80, "period_org": 34},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 100, "period_org": 10},
    #                 {"t0": 0.4, "length_org": 80, "period_org": 34},
    #                 {"proba_add": .3}],
    #                [{"t0": 0, "length_org": 100, "period_org": 10, "beta": .25, "max": 0.2},
    #                 {"t0": 0.4, "length_org": 80, "period_org": 34},
    #                 {"proba_add": .3}],
    #                [{"t0": 0, "length_org": 100, "period_org": 10, "beta": .01, "max": 0.2},
    #                 {"t0": 0.4, "length_org": 80, "period_org": 34},
    #                 {"proba_add": .3}]]

    # comb_params = [[{"t0": 0, "length_org": 100, "length_pm": .3, "period_org": 10, "period_pm": .1},
    #                 {"t0": 0.4, "length_org": 80, "length_pm": .3, "period_org": 34, "period_pm": .1},
    #                 {"proba_add": 0.}],
    #                [{"t0": 0, "length_org": 100, "length_pm": .3, "period_org": 10, "period_pm": .1},
    #                 {"t0": 0.4, "length_org": 80, "length_pm": .3, "period_org": 34, "period_pm": .1},
    #                 {"proba_add": .3}],
    #                [{"t0": 0, "length_org": 100, "length_pm": .3, "period_org": 10, "period_pm": .1, "beta": .25,
    #                "max": 0.2},
    #                 {"t0": 0.4, "length_org": 80, "length_pm": .3, "period_org": 34, "period_pm": .1},
    #                 {"proba_add": .3}],
    #                [{"t0": 0, "length_org": 100, "length_pm": .3, "period_org": 10, "period_pm": .1, "beta": .1,
    #                "max": 0.2},
    #                 {"t0": 0.4, "length_org": 80, "length_pm": .3, "period_org": 34, "period_pm": .1},
    #                 {"proba_add": .3}]]

    # for ci, cparams in enumerate(comb_params):
    #     seq, dets = generateSequences(cparams)

    # fparams = {"filename": "/home/egalbrun/TKTL/misc/itrami/per-pat/data/traces/prepared/trace_bugzilla_1_data.dat",
    #                  "timestamp": False, "events": ["2269:X"], "max_len": 700, "min_len": 20}
    # fparams = {"filename": "/home/egalbrun/TKTL/misc/itrami/per-pat/data/traces/prepared/trace_kptrace_3zap_0_data
    # .dat",
    #                "timestamp": False, "events": ["*"], "min_len": 3000, "max_len": 10000}
    fparams = {"filename": "/home/egalbrun/TKTL/misc/itrami/per-pat/data/traces/prepared/trace_kptrace_3zap_1_data.dat",
               "timestamp": False, "events": ["*"], "min_len": 30, "max_len": 3000}
    # fparams = {"filename": "/home/egalbrun/TKTL/misc/itrami/per-pat/data/traces/prepared/trace_bugzilla_1_data.dat",
    #                "timestamp": False, "events": ["*"], "min_len": 30, "max_len": 3000}

    # seqs = readSequence(fparams)

    seq = []
    for i in range(15):
        seq.append(i * 400 + numpy.arange(0, 20, 2))
    seqs = {0: numpy.hstack(seq)}

    for ci, seq in seqs.items():
        dets = None

        ds = DataSequence({"a": seq})
        print("------------")
        if len(seq) < 30:
            print("SEQUENCE %s: (%d)\t %s" % (ci, len(seq), seq))
        else:
            print("SEQUENCE %s: (%d)" % (ci, len(seq)))
        data_details = ds.getDetails()

        results = run_test(seq, "a", data_details, dets)
        for result in results:
            nb_cycl = len([c for c in result["cycles"] if c.get("cp") > 0])
            nb_res = numpy.sum([len(c["occs"])
                                for c in result["cycles"] if c.get("cp") <= 0])

            print("%s:\tCL=%f nC=%d nR=%d RT=%f" %
                  (result["meth"], result["CL"], nb_cycl, nb_res, result["RT"]))

            if len(result["cycles"]) > 0 and result["RT"] > 0:
                for c in result["cycles"]:
                    if c.get("cp") > 0:
                        print("\tp=%d\tr=%d\t%s" %
                              (c.get("cp"), len(c["occs"]), c["occs"]))
