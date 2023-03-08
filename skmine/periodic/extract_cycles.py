import datetime
import pdb

import numpy

from .class_patterns import computeLengthCycle, computeLengthResidual, computePeriod, \
    cost_triple, cost_one

DYN_BLOCK_SIZE = 100
STEP_SIZE = 80


def recover_splits_rec(spoints, ia, iz, depth=0, singletons=True):
    """
    Extract segments boundaries from dynamic programming output

    Parameters
    ----------
    spoints
    ia
    iz
    depth
    singletons

    Returns
    -------

    """
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
    """
    FIXME : to be explained

    Parameters
    ----------
    splits
    adj_splits

    Returns
    -------

    """
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
    """
    FIXME : to be explained

    Parameters
    ----------
    occs
    alpha
    data_details
    residuals

    Returns
    -------

    """
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
    for _, s in enumerate(splits):
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


def extract_cycles_fold(occs, alpha, data_details, bound_dE, eff_trip, eff_chain, max_p=None):
    """
    FIXME : to be explained

    Parameters
    ----------
    occs
    alpha
    data_details
    bound_dE
    eff_trip
    eff_chain
    max_p

    Returns
    -------

    """
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
    """
    FIXME : to be explained

    Parameters
    ----------
    occs
    alpha
    data_details
    bound_dE
    eff_trip
    eff_chain
    offset
    max_p

    Returns
    -------

    """
    centers = list(range(len(occs) - 1, 0, -1))
    pairs_chain_test = {}
    pairs_chain_fwd = {}
    pairs_chain_bck = {}
    triples_tmp = {}

    while len(centers) > 0:
        i = centers.pop()

        # compute distance to occurrence before and after center position
        bef = occs[:i][::-1] - occs[i]
        aft = occs[i + 1:] - occs[i]
        if max_p is not None:
            aft = aft[aft < max_p]
            bef = bef[bef > -max_p]
        seen_ps = []
        ia, ib = (0, 0)
        while ia < len(aft) and ib < len(bef):
            if ib < len(bef) - 1 and numpy.abs(aft[ia] + bef[ib]) > numpy.abs(aft[ia] + bef[ib + 1]):
                ib += 1
            elif ia < len(aft) - 1 and numpy.abs(aft[ia] + bef[ib]) > numpy.abs(aft[ia + 1] + bef[ib]):
                ia += 1
            else:
                deltaE = numpy.abs(aft[ia] + bef[ib])

                if deltaE < bound_dE:
                    dp = numpy.min([aft[ia], -bef[ib]])
                    dp_alt = numpy.max([aft[ia], -bef[ib]])
                    ipp = 0
                    while 0 <= ipp < len(seen_ps):
                        if dp == 2 * seen_ps[ipp]:
                            seen_ps.append(dp)
                            ipp = -seen_ps[ipp]
                        elif dp_alt == 2 * seen_ps[ipp]:
                            seen_ps.append(dp_alt)
                            ipp = -seen_ps[ipp]
                        else:
                            ipp += 1
                    if ipp >= 0:
                        costC = cost_triple(data_details, alpha, dp, deltaE)

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
            triples_tmp.update(
                dict([(k, v) for (k, v) in pairs_chain_test.items() if k[1] < i]))
            pairs_chain_test = dict(
                [(k, v) for (k, v) in pairs_chain_test.items() if k[1] >= i])

    triples_tmp.update(pairs_chain_test)

    chains = []

    while len(pairs_chain_fwd) > 0:
        kf = min(pairs_chain_fwd.keys())
        current = [pairs_chain_bck[kf][-1], kf[0], kf[1]]
        while (current[-2], current[-1]) in pairs_chain_fwd:
            nxt = pairs_chain_fwd.pop((current[-2], current[-1]))
            if (current[-1], nxt[-1]) in triples_tmp:
                del triples_tmp[(current[-1], nxt[-1])]
            current.append(nxt[-1])

        occs_chain = [occs[c] for c in current]
        prd = computePeriod(occs_chain)
        cost = computeLengthCycle(
            data_details, {"p": prd, "alpha": alpha, "occs": occs_chain})
        if (eff_chain < 0 and cost < (-eff_chain) * (len(occs_chain) - 1)) or (cost / len(occs_chain) < eff_chain):
            chains.append({"alpha": alpha, "p": prd, "cost": cost,
                           "pos": current, "occs": occs_chain, "uncov": set(current)})

    triples = [(v[0], v[1], v[-1], (v[-2] + offset, k[0] + offset, k[1] + offset)) for (k, v)
               in triples_tmp.items() if (eff_trip < 0 and v[-1] < -2 * eff_trip) or (v[-1] / 3. < eff_trip)]

    return chains, triples


