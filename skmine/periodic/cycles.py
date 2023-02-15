"""Periodic pattern mining with a MDL criterion"""
# Authors: Rémi Adon <remi.adon@gmail.com>
#          Esther Galbrun <esther.galbrun@inria.fr>
#
# License: BSD 3 clause

import warnings
import numpy as np
import pandas as pd

from pyroaring import BitMap as Bitmap

from ..base import BaseMiner, DiscovererMixin, MDLOptimizer
from sklearn.base import BaseEstimator, TransformerMixin

from .class_patterns import PatternCollection
from .run_mine import mine_seqs
import datetime

log = np.log2

INDEX_TYPES = (
    pd.DatetimeIndex,
    pd.RangeIndex,
    pd.Int64Index,
)


def _remove_zeros(numbers: pd.Series):
    n = 0
    while (numbers % 10 == 0).all():
        numbers //= 10
        n += 1
    return numbers, n


def _reconstruct(start, period, dE):
    """
    Reconstruct occurrences,
    starting from `start`, and
    correcting `period` with a delta for all deltas in `dE`,
    `len(dE)` occurrences are reconstructed

    Parameters
    ----------
    start: int or datetime
        starting point for the event
    period: int or timedelta
        period between two occurrences
    d_E: np.array of [int|timedelta]
        inters occurrences deltas
    """
    occurrences = [start]
    current = start
    for d_e in dE:
        e = current + period + d_e
        occurrences.append(e)
        current = e
    return occurrences


class PeriodicCycleMiner(TransformerMixin, BaseEstimator):
    # (BaseMiner, MDLOptimizer, DiscovererMixin):
    """
    Mining periodic cycles with a MDL Criterion

    PeriodicCycleMiner is an approach to mine periodic cycles from event logs
    while relying on a Minimum Description Length (MDL) criterion to evaluate
    candidate cycles. The goal here is to extract a set of cycles that characterizes
    the periodic structure present in the data

    A cycle is defined a 5-tuple of of the form
        .. math:: \\alpha, r, p, \\tau, E

    Where

    - :math:`\\alpha` is the `repeating event`
    - :math:`r` is the number of repetitions of the event, called the `cycle length`
    - :math:`p` is the inter-occurrence distance, called the `cycle period`
    - :math:`\\tau` is the index of the first occurrence, called the `cycle starting point`
    - :math:`E` is a list of :math:`r - 1` signed integer offsets, i.e `cycle shift corrections`

    Examples
    --------
    >>> from skmine.periodic import PeriodicCycleMiner
    >>> import pandas as pd
    >>> S = pd.Series("ring_a_bell", [10, 20, 32, 40, 60, 79, 100, 240])
    >>> pcm = PeriodicCycleMiner().fit(S)
    >>> pcm.discover()
       start  length  period       cost                             residuals 	event
    0     10       3      11  23.552849 {(240, 0), (10, 0), (32, 0)}            [ring_a_bell]
    1     40       4 	  20  29.420668 {(20, 0), (240, 0), (10, 0), (32, 0)} 	[ring_a_bell]

    References
    ----------
    .. [1]
        Galbrun, E & Cellier, P & Tatti, N & Termier, A & Crémilleux, B
        "Mining Periodic Pattern with a MDL Criterion"
    """

    def __init__(self):
        self.miners_ = dict()
        self.is_datetime_ = None
        self.n_zeros_ = 0
        self.alpha_groups = {}
        self.cycles = None
        self.data_details = None
        self.auto_time_scale = True

    def fit(self, S, complex=True, auto_time_scale=True):
        """fit PeriodicCycleMiner on data logs

        This generate new candidate cycles and evaluate them.
        Residual occurrences are stored as an internal attribute,
        for later reconstruction (MDL is lossless)

        Parameters
        -------
        S: pd.Series
            logs, represented as a pandas Series
            This pandas Series must have an index of type in
            (pd.DatetimeIndex, pd.RangeIndex, pd.Int64Index)

        complex: boolean
            True : compute complex pattern with horizontal and vertical combinations.
            False: compute only simple cycles.

        auto_time_scale: boolean
            True : preprocessing on time data index. Calcul automaticaly the time scale for mining cycles by removing extra zeros on time index.
            False: no preprocessing on time data index
        """
        self.auto_time_scale = auto_time_scale

        if not isinstance(S, pd.Series):
            raise TypeError("S must be a pandas Series")

        if not isinstance(S.index, INDEX_TYPES):
            raise TypeError(
                f"S must have an index with a type amongst {INDEX_TYPES}")

        self.is_datetime_ = isinstance(S.index, pd.DatetimeIndex)

        if S.index.duplicated().any():
            # warnings.warn("found duplicates in input sequence, removing them")
            S = S.groupby(S.index).first()

        S = S.copy()

        if self.auto_time_scale:
            S.index, self.n_zeros_ = _remove_zeros(S.index.astype("int64"))

        # TODO : do this in SingleEventCycleMiner?

        # n_occs_tot = S.shape[0]
        # S.index = S.index.astype(int)

        self.alpha_groups = S.groupby(S.values).groups

        #  ========================================================================================
        #  ****************************************************************************************
        #  TODO : Connection with Esther routine here :

        cpool, data_details, pc = mine_seqs(dict(self.alpha_groups),
                                            fn_basis="test", complex=complex)

        print("cpool", cpool)

        self.data_details = data_details
        self.miners_ = pc

        out_str, pl_str = self.miners_.strDetailed(self.data_details)
        print(out_str)

        # print("\n\n\n\n *********** Candidates *************\n\n\n\n")
        # print("nbCandidates()", cpool.nbCandidates())

        # print("\n\n *********** pc *************\n\n")
        # print("__len__ pc", pc.__len__())
        # print("\n\n nbPatternsByType", pc.nbPatternsByType())
        # print("setPatterns", pc.setPatterns( patterns=[]))
        # print("addPatterns", pc.addPatterns(patterns))
        # print("\n\n getPatterns", pc.getPatterns()[0:10])
        # print("\n\ngetCoveredOccs", pc.getCoveredOccs()[0:10])
        # print("getUncoveredOccs", pc.getUncoveredOccs(data_seq))
        # print("getNbUncoveredOccsByEv", pc.getNbUncoveredOccsByEv(data_seq))
        # print("\n\ngetOccLists", pc.getOccLists()[0:10])
        # print("codeLength", pc.codeLength(data_seq))

        # pid = 0
        # pids = [0, 1]
        # cid = 0
        # cid = 0
        # # event (α), length (r), period (p), starting point (τ) and shift corrections (E)
        # print("isNewPid(pid)", cpool.isNewPid(pid))
        # print("areNewPids(pids)", cpool.areNewPids(pids))
        # print("getPidsForCid(cid)", cpool.getPidsForCid(cid))
        # print("getCidsForMinorK(mK)", cpool.getCidsForMinorK(mK))
        # print("getCandidates()", cpool.getCandidates())
        # print("getNewKNum(nkey)", cpool.getNewKNum(nkey))
        # print("getNewPids(nkey)", cpool.getNewPids(nkey))
        # print("getNewCids(nkey)", cpool.getNewCids(nkey))
        # print("getNewMinorKeys(nkey)", cpool.getNewMinorKeys(nkey))
        # print("nbNewCandidates()", cpool.nbNewCandidates())
        # print("nbMinorKs()", cpool.nbMinorKs())
        # print("nbCandidates()", cpool.nbCandidates())
        # print("nbProps()", cpool.nbProps())
        # print("getPropMat()", cpool.getPropMat())
        # print("getProp(pid)", cpool.getProp(pid))

        #  ****************************************************************************************
        #  ========================================================================================

        # miners = {
        #     k: SingleEventCycleMiner(
        #         self.max_length, self.keep_residuals, n_occs_tot)
        #     for k in alpha_groups.keys()
        # }
        # self.miners_ = {k: miners[k].fit(v) for k, v in alpha_groups.items()}

        # for (event, miner,) in self.miners_.items():
        #     # print("event", type(event), event)
        #     # print("miner", type(miner), miner)
        #     # print("miner.cycles_ :\n", miner.cycles_)
        #     # print("miner.residuals_ :\n", miner.residuals_)
        #     # print("miner.max_length :\n", miner.max_length)
        #     # print("miner.keep_residuals :\n", miner.keep_residuals)
        #     # print("miner.n_occs_tot :\n", miner.n_occs_tot)
        #     # print("miner._dS :\n", miner._dS)
        #     # print("miner.k :\n", miner.k)
        #     # print("miner.tid_pad :\n", miner.tid_pad)

        #     if "tids" in miner.cycles_.columns:
        #         # FIXME: this is highly inefficient

        #         miner.cycles_.tids = miner.cycles_.tids.map(
        #             lambda tids: Bitmap(
        #                 np.searchsorted(S.index, alpha_groups[event][tids])
        #             )
        #         )

        return self

    # evaluate = SingleEventCycleMiner.evaluate

    def discover(self, dE_sum=True):
        """Return cycles as a pandas DataFrame, with 3 columns,
        with a 2-level multi-index: the first level mapping events,
        and the second level being positional

        Parameters
        -------
        dE_sum: boolean
            True : returm a columns "dE" with the sum of the errors 
            False: returm a columns "dE" with the full list of errors.  


        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  ======================================
                start       when the cycle starts
                length      number of occurrences in the event
                period      inter-occurrence delay
                dE          shift corrections
                cost        MDL cost
                ==========  ======================================

            Examples
            --------
            >>> from skmine.periodic import PeriodicCycleMiner
            >>> import pandas as pd
            >>> S = pd.Series("ring_a_bell", [10, 20, 32, 40, 60, 79, 100, 240])
            >>> pcm = PeriodicCycleMiner().fit(S)
            >>> pcm.discover()
            start  length  period       cost                                residuals 	        event              dE
            0     10       3      11  23.552849 {(240, 0), (10, 0), (32, 0)}            [ring_a_bell]   [0, 0, -1, 1]
            1     40       4 	  20  29.420668 {(20, 0), (240, 0), (10, 0), (32, 0)} 	[ring_a_bell]   [0, -1, 1]
        """

        global_stat_dict, patterns_list_of_dict = self.miners_.output_detailed(
            self.data_details)

        # print(pd.DataFrame([global_stat_dict]))

        if not patterns_list_of_dict:
            return pd.DataFrame()  # FIXME

        self.cycles = pd.DataFrame(patterns_list_of_dict)

        if self.auto_time_scale:
            self.cycles.loc[:, ["t0", "period_major"]
                            ] *= 10 ** self.n_zeros_

            # self.cycles.loc[:, "dE"] = self.cycles.dE.map(
            #     np.array) * (10 ** self.n_zeros_)

            # self.cycles.loc[:, "dE"] = self.cycles.loc[:, "dE"] * (10 ** self.n_zeros_)
            # disc_print = self.cycles.copy()
            if self.is_datetime_:
                self.cycles.loc[:, "t0"] = self.cycles.t0.astype(
                    "datetime64[ns]")
                self.cycles.loc[:, "period_major"] = self.cycles.period_major.astype(
                    "timedelta64[ns]")

        return self.cycles

    def reconstruct(self, *patterns_id):
        """Reconstruct the original occurrences from the current cycles.
        Residuals will also be included, as the compression scheme is lossless

        Denoting as :math:`\sigma(E)` the sum of the shift corrections for a cycle
        :math:`C`, we have

        .. math::
            \Delta(C)=(r-1) p+\sigma(E)

        Returns
        -------
        pd.Series
            The reconstructed dataset

        Notes
        -----
        The index of the resulting pd.Series will not be sorted
        """

        reconstruct_list = []

        map_ev = self.data_details.getNumToEv()

        if patterns_id == ():
            patterns_id = range(len(self.miners_.getPatterns()))
        else:
            patterns_id = patterns_id[0]

        for pattern_id in patterns_id:
            (p, t0, E) = self.miners_.getPatterns()[pattern_id]

            occsStar = p.getOccsStar()
            Ed = p.getEDict(occsStar, E)
            occs = p.getOccs(occsStar, t0, Ed)

            for k, occ in enumerate(occs):

                dict_ = {}
                dict_['time'] = (
                    occ)*10**self.n_zeros_ if self.auto_time_scale else (occ)

                dict_["event"] = map_ev[occsStar[k][1]]
                reconstruct_list.append((dict_))

        reconstruct_pd = pd.DataFrame(reconstruct_list)

        if self.is_datetime_:
            reconstruct_pd['time'] = reconstruct_pd['time'].astype(
                "datetime64[ns]")
            reconstruct_pd['time'] = reconstruct_pd['time'].astype("str")

        return reconstruct_pd

    # def generate_candidates(self, S):
    #     """
    #     Parameters
    #     ----------
    #     S: pd.Index or numpy.ndarray
    #         Series of occurrences for a specific event

    #     Returns
    #     -------
    #     dict[object, list[np.ndarray]]
    #         A dict, where each key is an event and each value a list of batch of candidates.
    #         Batches are sorted in inverse order of width,
    #         so that we consider larger candidate cycles first.
    #     """
    #     # TODO only for InteractiveMode

    def get_residuals(self, *patterns_id):
        """Get the residual events, i.e events not covered by any cycle

        It is the complementary function to `discover`

        Returns
        -------
        pd.Series
            residual events
        """

        if patterns_id == ():
            patterns_id = range(len(self.miners_.getPatterns()))
        else:
            patterns_id = patterns_id[0]

        map_ev = self.data_details.getNumToEv()
        residuals = self.miners_.getUncoveredOccs(self.data_details)
        residuals_transf_list = []

        for res in residuals:
            dict_ = {}
            dict_["time"] = res[0] * \
                10**self.n_zeros_ if self.auto_time_scale else res[0]
            dict_["event"] = map_ev[res[1]]
            residuals_transf_list.append(dict_)

        residuals_transf_pd = pd.DataFrame(residuals_transf_list)

        if self.auto_time_scale and self.is_datetime_:
            residuals_transf_pd['time'] = residuals_transf_pd['time'].astype(
                "datetime64[ns]")
            residuals_transf_pd['time'] = residuals_transf_pd['time'].astype(
                "str")

        reconstruct_ = self.reconstruct(patterns_id)
        reconstruct_all = self.reconstruct()

        reconstruct_ = self.reconstruct(patterns_id)
        reconstruct_all = self.reconstruct()

        complementary_reconstruct = reconstruct_all[~reconstruct_all.isin(
            reconstruct_)].dropna()

        if pd.merge(reconstruct_all, residuals_transf_pd, how='inner').empty:
            residuals_transf_pd = pd.concat(
                [residuals_transf_pd, complementary_reconstruct], ignore_index=True)
        else:
            warnings.warn(
                "resudials and complementary of reconstruct have common patterns")
            residuals_transf_pd = pd.concat(
                [residuals_transf_pd, complementary_reconstruct], ignore_index=True)

        return residuals_transf_pd
