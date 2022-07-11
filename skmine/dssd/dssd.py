from collections import defaultdict
from distutils.file_util import write_file
import math
import os
import time
import logging
from typing import List, Dict
from pandas import DataFrame

from .refinement_operators import RefinementOperator, RefinementOperatorOfficial
from .utils import _min_max_avg_quality_string, sort_subgroups, remove_duplicates, subgroup, diff_items_count, func_get_quality, subgroup, subgroups_to_csv
from .subgroup import Subgroup
from .description import Description
from .custom_types import ColumnType, FuncCover, FuncQuality
from .selection_strategies import FixedDescriptionBasedSelectionStrategy, SelectionStrategy
from .quality_measures import QualityMeasure

# Create a custom logger
logger = logging.getLogger("dssd")


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

        # add the cover computing function here and also deem the quality function to be completely dependant on what is inside the candidate cover at the moment of the call
        # so if the cover should be updated we make sure to do it before calling the quality function
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
    List[Subgroup]: The same list received in argument is returned for convenience purposes
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


# Actual mining function


def setup_logging(stdoutfile: str, level = logging.DEBUG):
    """Setup the logger to be used by the dssd algorithm and other parts.
    This function should be called only once, in the dssd algorithm otherwise, weird things might happen
    """
    # Create handlers
    c_handler = logging.StreamHandler()
    file_debug_handler = logging.FileHandler(f"{stdoutfile}.debug.log")
    file_info_handler = logging.FileHandler(f"{stdoutfile}.info.log")
    c_handler.setLevel(logging.DEBUG)
    file_debug_handler.setLevel(logging.DEBUG)
    file_info_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_format = logging.Formatter('%(asctime)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s: %(message)s')
    c_handler.setFormatter(c_format)
    file_info_handler.setFormatter(f_format)
    file_debug_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(file_info_handler)
    logger.addHandler(file_debug_handler)

    logger.setLevel(level)

    return logger

def mine(
    k: int,
    j: int = None,
    max_depth: int = 0,
    beam_width: int = None,
    quality_measure: QualityMeasure = None,
    selector: SelectionStrategy = FixedDescriptionBasedSelectionStrategy(),
    post_selector: SelectionStrategy = None,
    refinement_operator: RefinementOperator = None,
    output_folder: str = "",
    save_intermediate_results: bool = True,
    skip_phase2: bool = False,
    skip_phase3: bool = False) -> List[Subgroup]:
    """
    Mine and return mined subgroups

    Parameters
    ----------
    max_depth: int
        Maximum number of conditions per subgroup description
    k: int
        The number of subgroups to be returned as a result of the experiment
    j: int 
        The maximum number of subgroups to be kept in memory at all time during the process. Defaults to math.inf.
    beam_width: int, default=k
        The beam width to use during phase 1.
    quality_measure: QualityMeasure, default=None
        An implementation of a quality measure to be used.
        Notes: This is where the target attributes go
        The quality measure is expected to operate on a projection of the dataset on the target attributes
    selector: SelectionStrategy, default=FixedDescriptionBasedSelectionStrategy()
        An implementation of a selection strategy to be used (during phase 1).
    post_selector: SelectionStrategy, default=selection_strategy
        An implementation of a selection strategy to be used (during phase 3).
    refinement_operator: RefinementOperator, default=RefinementOperatorOfficial()
        An implementation of the refinement operator to be used.
        Remember this is where the desccriptive attributes go, the minimum coverage, min quality, numeric discretisation cut points 
        techniques, etc.
        The operator is expected to operate on a projection of the dataset on the descriptive attributes
        The quality and cover computing function are automatically filled in later by the preparation functions
    output_folder: str, default=""
        The output folder where to store (intermediate) results
    save_intermediate_results: bool, default=True
        whether or not to save intermediate results at each depth of the search phase
    skip_phase2: bool, default=False
        Whether or not to skip phase 2. Defaults to False
    skip_phase3: bool, default=False
        Whether or not to skip phase 3. Defaults to False

    Returns
    -------
    List[Subgroup]

    References
    ----------
    [1] Page 224-225
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    # write a string version of all the arguments received to a config file, helpful to later remember what config yielded what result
    local_args = locals()
    function_args = "\n".join([f'{k}={v}' for k,v in local_args.items()])
    write_file(f"{output_folder}/dssd_params.conf", [function_args])


    logger = setup_logging(f"{output_folder}/stdout")
    # a wrapper function that computes subgroups quality
    quality_func: FuncQuality = lambda c: quality_measure.compute_quality(quality_measure._df.loc[c.cover])
    # an "optimized" function to compute the cover of a subgroup, it is optimized cause it uses its parent cover as a base and only checks last added condition
    cover_func_optimized: FuncCover = lambda c: subgroup(ro.df.loc[c.parent.cover], c.description, True)
    # normal unoptimized wrapper function for computing the cover of a subgroup
    cover_func_non_optimized: FuncCover = lambda c: subgroup(ro.df, c.description, False)
    
    ro = refinement_operator
    # fill in the fields of the refinement operator
    ro.cover_func = cover_func_optimized
    ro.quality_func = quality_func

    logger.info(f"Phase 1: Mining {j} subgroups each having at most {max_depth} conditions each")

    # initialize a beam with a base candidate subgroup, containing all elements of the initial dataset
    beam = [Subgroup(Description([]), 0, cover=ro.df.index, parent=None)]
    result: List[Subgroup] = []
    start_time = time.time()
    depth = 1
    while depth <= max_depth:
        candidates: List[Subgroup] = []
        logger.info(f"Generating depth {depth} candidates")
        for cand in beam:
            # all valid candidates that extend current candidate's pattern with one condition
            candidates += ro.refine_candidate(cand, [])

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
            apply_dominance_pruning(cand, quality_func, cover_func_non_optimized)

        # remove duplicates that may have occured due to the pruning process
        logger.info(f"Removing duplicates...")
        result = remove_duplicates(result)
        logger.info(f"{len(result)} remaining after candidates deduplication...")
        if save_intermediate_results: write_file(f"{output_folder}/stats2-after-duplicates-removal.csv", [subgroups_to_csv(result)])

    # final candidates selection / post selection phase
    if not skip_phase3:
        logger.info(f"Phase 3: Post selecting {k} candidates out of {len(result)}...")
        result = post_selector.select(result, beam_width=k, beam=[])
        sort_subgroups(result)

    logger.info(f"Total time taken = {time.time() - start_time} seconds\n")
    logger.info(f"Final selection\n{len(result)} candidate(s)\n{_min_max_avg_quality_string(result, ' ')}")
    logger.info(f"Results stored in the folder: {os.getcwd()}/{output_folder}")
    write_file(f"{output_folder}/stats3-final-results.csv", [subgroups_to_csv(result)])
    return result
