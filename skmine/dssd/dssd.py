from distutils.file_util import write_file
import time
import logging
from typing import List
from .refinement_operators import RefinementOperator, RefinementOperatorImpl
from .utils import _min_max_avg_quality_string, sort_subgroups, remove_duplicates, subgroup, subgroup, subgroups_to_csv, dummy_logger
from .subgroup import Subgroup
from .description import Description
from .custom_types import  FuncCover, FuncQuality
from .selection_strategies import Desc, SelectionStrategy
from .quality_measures import QualityMeasure


def apply_dominance_pruning(candidate: Subgroup, quality_func: FuncQuality, cover_func: FuncCover):
    """
    Apply dominance pruning to the candidate which consits of removing 
    conditions which absence doesn't descrease the subgroup quality

    Parameters
    ----------
    candidate: Subgroup
        The subgroup to prune
    quality_func: FuncQuality
        A function to be used to re-compute subgroup quality while some 
        of its conditions are being removed 
    cover_func: FuncCover
        A function to be used to re-compute subgroup cover in order 
        to have accurate cover while computing quality
    
    References
    ----------
    [1] Page 226
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    
    # consider the conditions in each subgroup one by one and if removing a condition from a subgroup does not decrease its quality remove that condition permanently
    highest_quality = candidate.quality
    highest_cover = candidate.cover
    for (i, condition) in enumerate(candidate.description.conditions.copy()):
        # remove the condition from the candidate
        candidate.description.conditions.remove(condition)

        candidate.cover = cover_func(candidate).index
        new_quality = quality_func(candidate)
        if new_quality < highest_quality:
            # Add conditions back cause removing this condition decreased the quality of the candidate
            candidate.description.conditions.insert(i, condition)
            candidate.cover = highest_cover
        else: # removing this condition doesn't negatively affect quality so permanently remove it and update the candidate quality to the newly computed one
            highest_quality = new_quality
            candidate.quality = new_quality
            highest_cover = candidate.cover
    

def update_topk(result: List[Subgroup], candidate: Subgroup, max_size: int = 0):
    """
    Insert the new candidate to keep the result list sorted descending with at most the maximum specified size

    Parameters
    ----------
    result: List[Subgroup]
        The destination list where the candidate is to be inserted
    candidate: Subgroup
        The candidate to be inserted
    max_size: int, default=0
        The maximum size allowed for the result list

    Returns
    -------
    List[Subgroup]:
        The same list received in argument is returned for convenience purposes
    """
    
    # highly inspired by the bisect insort method in python 3.8
    def __insort(a: List, x, lo=0, hi=None, key=None):
        hi = hi if hi is not None else len(result)
        _x = key(x)
        while lo < hi:
            mid = (lo + hi) // 2
            if _x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
        a.insert(lo, x)

    # Insert the candidate into the result list in a way to keep the result list sorted descending
    __insort(result, candidate, key= lambda c: -c.quality)
    
    if len(result) > max_size:
        result.pop()
    return result


def mine(
    k: int,
    j: int,
    max_depth: int = 3,
    beam_width: int = None,
    quality: QualityMeasure = None,
    selector: SelectionStrategy = Desc(),
    post_selector: SelectionStrategy = None,
    ref_op: RefinementOperator = RefinementOperatorImpl(),
    output_folder: str = "",
    save_intermediate_results: bool = True,
    save_result: bool = True,
    skip_phase2: bool = False,
    skip_phase3: bool = False,
    return_pool: bool = False,
    sort_final_result: bool = False,
    logger: logging.Logger = dummy_logger) -> List[Subgroup]:
    """
    Mine and return mined subgroups

    Parameters
    ----------
    k: int
        The number of subgroups to be returned as a result of the experiment
    j: int 
        The maximum number of subgroups to be kept in memory at all time during the process
    max_depth: int, default=3
        Maximum number of conditions per subgroup description
    beam_width: int, default=k
        The beam width to use during phase 1
    quality: QualityMeasure, default=None
        An implementation of a quality measure to be used. The quality measure uses a projection of the dataset on the target attributes.
    selector: SelectionStrategy, default=Desc()
        An implementation of a selection strategy to be used (during phase 1).
    post_selector: SelectionStrategy, default=selector
        An implementation of a selection strategy to be used (during phase 3).
    ref_op: RefinementOperator, default=RefinementOperatorImpl()
        An implementation of the refinement operator to be used. 
        Notes: The operator is expected to operate on a projection of the dataset on the descriptive attributes, so this is where the descriptive attributes go and minimum coverage, min quality, numeric discretisation cut points variables are exploited.
        (The quality and cover computing function are automatically filled in later by the algorithm)
    output_folder: str, default=""
        The output folder where to store (intermediate) results and any other files created by the algorithm
    save_intermediate_results: bool, default=True
        whether or not to save intermediate results at each depth of the search phase
    save_result: bool, default=False
        whether or not to save the actual result of the mining process
    skip_phase2: bool, default=False
        Whether or not to skip phase 2
    skip_phase3: bool, default=False
        Whether or not to skip phase 3
    sort_final_result: bool, default=False
        Whether or not to sort the candidates quality descending during phase 3 after post selection.
    logger: Logger, default=dummy_logger
        Logger for the dssd algorithm and other parts

    Returns
    -------
    List[Subgroup] | (Tuple[List[Subgroup], List[Subgroup]]):
        Return the actual result and the pool used for final seleciton if return_pool is True 


    References
    ----------
    [1] Page 224-225
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    # write a string version of all the arguments received to a config file, helpful to later remember what config yielded what result
    beam_width = beam_width if beam_width is not None else k
    function_args = "\n".join([f'{k}={v}' for k,v in locals().items()])
    if save_result and output_folder != "": write_file(f"{output_folder}/dssd_params.conf", [function_args])

    # fill in the fields of the refinement operator
    ref_op.cover_func = lambda c: subgroup(ref_op.df.loc[c.parent.cover], c.description, True)
    ref_op.quality_func = quality.quality_from_subgroup

    logger.info(f"Phase 1: Mining {j} subgroups each having at most {max_depth} conditions each")

    # initialize a beam with a base candidate subgroup, containing all elements of the initial dataset
    beam = [Subgroup(Description([]), 0, cover=ref_op.df.index, parent=None)]
    result: List[Subgroup] = []
    start_time = time.time()
    depth = 1
    while depth <= max_depth:
        candidates: List[Subgroup] = []
        logger.info(f"Generating depth {depth} candidates")
        for cand in beam:
            # all valid candidates that extend current candidate's pattern with one condition
            candidates += ref_op.refine_candidate(cand, [])

        if len(candidates) == 0:
            logger.warning(f"depth={depth} : Generated 0 candidates... Stopping exploration phase\n")
            break
        logger.info(f"depth={depth} : generated {len(candidates)} candidates")
        logger.info(f"depth={depth} : {_min_max_avg_quality_string(candidates, ' ')}")
        for cand in candidates:
            update_topk(result, cand, j)
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-generated-candidates.csv", [subgroups_to_csv(candidates)])

        beam = selector.select(candidates, beam_width=beam_width, beam = [])
        logger.info(f"depth={depth} : selected {len(beam)} candidates\n")
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-selected-candidates.csv", [subgroups_to_csv(beam)])
        depth += 1

    # prune each candidate's pattern to only keep the actually useful conditions
    if save_intermediate_results: write_file(f"{output_folder}/stats1-after-mining-and-before-pruning.csv", [subgroups_to_csv(result)])
    if not skip_phase2:
        logger.info(f"Phase 2: Dominance pruning & deduplication")
        logger.info(f"Dominance pruning {len(result)} candidates...")
        for cand in result:
            apply_dominance_pruning(cand, quality.quality_from_subgroup, lambda c: subgroup(ref_op.df, c.description, False))

        # remove duplicates that may have occured due to the pruning process
        logger.info(f"Removing duplicates...")
        result = remove_duplicates(result)
        logger.info(f"{len(result)} remaining after candidates deduplication...")
        if save_intermediate_results: write_file(f"{output_folder}/stats2-after-duplicates-removal.csv", [subgroups_to_csv(result)])

    # final candidates selection / post selection phase
    if return_pool: 
        pool = result.copy()
    if not skip_phase3:
        logger.info(f"Phase 3: Post selecting {k} candidates out of {len(result)}...")
        result = post_selector.select(result, beam_width=k, beam=[])
        if sort_final_result: sort_subgroups(result)

    logger.info(f"Total time taken = {time.time() - start_time} seconds\n")
    logger.info(f"Final selection\n{len(result)} candidate(s)\n{_min_max_avg_quality_string(result, ' ')}")
    if save_result: write_file(f"{output_folder}/stats3-final-results.csv", [subgroups_to_csv(result)])
    if return_pool:
        return result, pool
    return result
