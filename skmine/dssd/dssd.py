from bisect import insort
from collections import defaultdict
from distutils.file_util import write_file
import math
import os
import time
import logging
from typing import Any, DefaultDict, List, Dict
from pandas import DataFrame
from .table import Table
from .utils import sub_dict, min_max_avg_quality_string, sort_subgroups, remove_duplicates, subgroup, diff_items_count, func_get_quality, subgroup, to_csv
from .subgroup import Subgroup
from .description import Description
from .custom_types import ColumnType, FuncCover, FuncQuality
# Create a custom logger
logger = logging.getLogger("dssd")
# from __future__ import annotations


def fixed_size_description_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, min_diff_conditions: int = 2) -> List[Subgroup]:
    def is_candidate_diverse(candidate: Subgroup, beam: List[Subgroup], min_diff_conditions: int) -> bool:
        # return not any(c.quality == candidate.quality and diff_items_count(c.description.conditions, candidate.description.conditions) < min_diff_conditions for c in beam)
        """not working well cause of how the difference is computed, it should just say how differnt two lists are regardless of order otherwise that is not good and strange things can happen so change it back to be orderless"""
        return all(c.quality != candidate.quality or diff_items_count(c.description.conditions, candidate.description.conditions) >= min_diff_conditions for c in beam)

    logger.debug(f"desc: beam_width={beam_width}")
    selected_candidates_count = 0
    candidate_index = 0
    # Make sure candidates are ordered in quality descending order, this might not be required if we assume an ordering pre-condition
    sort_subgroups(candidates)
    
    while candidate_index < len(candidates) and selected_candidates_count < beam_width:
        candidate = candidates[candidate_index]
        candidate_diverse = is_candidate_diverse(candidate, beam, min_diff_conditions)
        if candidate_diverse:
            beam.append(candidate)
            selected_candidates_count += 1
            logger.debug(f"SELECTED CANDIDATE N°{selected_candidates_count}")
        candidate_index += 1
    return beam


def var_size_description_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, c: int, l: int) -> List[Subgroup]:
    def is_candidate_diverse(candidate: Subgroup, attributes_usage: DefaultDict[str, int], max_occ: int) -> bool:
        for cond in candidate.description.conditions:
            if attributes_usage[cond.attribute] >= max_occ: # discard this candidate as it has at least an attribute already overused
                return False
        return True

    def update_beam(beam: List[Subgroup], attributes_usage: Dict[str, int], candidate: Subgroup):
        beam.append(candidate)
        # Update the usage count of the attributes present in the selected candidate
        for cond in candidate.description.conditions:
            # realize that this can cause some attributes reach a usage count higher that max_occ as if 
            # a description contains multiple conditions on the same attribute and that attribute had 
            # just one more legal possible usage. Might wanna check for a way to aggregate the number 
            # of usages before processing to the diversity computing strategy
            attributes_usage[cond.attribute] += 1


    selected_candidates_count = 0
    candidate_index = 0
    # Make sure candidates are ordered in quality descending order
    sort_subgroups(candidates)
    # candidates = sorted(candidates, key = lambda c: c.quality, reverse=True)
    attributes_usage = defaultdict(int, {})
    max_occ = c * l
    logger.debug(f"var_desc: beam_width={beam_width}, max_occ={max_occ}")

    while candidate_index < len(candidates) and selected_candidates_count < beam_width:
        candidate = candidates[candidate_index]
        candidate_diverse = is_candidate_diverse(candidate, attributes_usage, max_occ)
        
        if candidate_diverse:
            selected_candidates_count += 1
            update_beam(beam, attributes_usage, candidate)
            logger.debug(f"SELECTED CANDIDATE {selected_candidates_count}")

        candidate_index += 1 

    logger.debug(f"attributes_usage={attributes_usage}")
    return beam


def multiplicative_weighted_covering_score(cand: Subgroup, selection: List[Subgroup], weight: float) -> float:
    """Compute and return the score for the current candidate based on how often the transactions 
    in its cover are already covered by the candidates in the selection
    Requirement:
        | cand.cover | > 0

    Args:
        cand (Candidate): the candidate to rank
        selection (List[Candidate]): the list of already selected candidates
        weight (float): the initial weight of every transaction

    Returns:
        float
    """
    if len(cand.cover) <= 0: 
        raise ValueError("Can not compute the score for a candidate which cover is empty")

    if not (0 < weight <= 1): 
        raise ValueError("Value needs be in (0, 1] ")

    result = 0.
    for transaction in cand.cover:
        count = 0
        # count how many times the current transaction/tuple is already present in the selection:  c(t, sel)
        for candidate in selection:
            if transaction in candidate.cover:
                count += 1

        # Add the weighted result; alpha ^ c(t, sel)
        result += pow(weight, count)
    return result / (1 if cand.cover.size == 0 else cand.cover.size)


def multiplicative_weighted_covering_score_smart(cand: Subgroup, weight: float, counts: Dict[int, int]) -> float:
    """Compute and return the score for the current candidate based on how often the transactions 
    in its cover are already covered by the candidates in the selection
    Requirement:
        | cand.cover | > 0

    Args:
        cand (Candidate): the candidate to rank
        selection (List[Candidate]): the list of already selected candidates
        weight (float): the initial weight of every transaction

    Returns:
        float
    """
    if len(cand.cover) <= 0: 
        raise ValueError("Can not compute the score for a candidate which cover is empty")

    if not (0 < weight <= 1): 
        raise ValueError("Value needs be in (0, 1] ")

    result = 0.
    for transaction in cand.cover:
        result += pow(weight, counts[transaction])
    return result / len(cand.cover)


def update_counts(cand: Subgroup, counts: Dict[int, int]):
    """Increase by 1 the counts of every tuple from the specified candidate"""
    for t in cand.cover:
        counts[t] += 1


def fixed_size_cover_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, weight: float) -> List[Subgroup]:
    counts = defaultdict(int, {})
    score: FuncQuality = lambda c: multiplicative_weighted_covering_score_smart(c, weight, counts) * c.quality

    # in case there are less candidates than the beam width
    # just retrun the candidates list
    if len(candidates) <= beam_width:
        beam.extend(candidates)
        return beam

    logger.debug(f"cover: beam_width={beam_width}")

    for i in range(beam_width):
        logger.debug(f"SELECTED CANDIDATE {i + 1}")
        max_scoring_candidate = max(candidates, key = score)
        # Select the the candidate with the highest score for beam_width times
        candidates.remove(max_scoring_candidate)
        beam.append(max_scoring_candidate)
        update_counts(max_scoring_candidate, counts)

    return beam


def var_size_cover_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, weight: float, fraction: float) -> List[Subgroup]:
    counts = defaultdict(int, {})
    score: FuncQuality = lambda c: multiplicative_weighted_covering_score_smart(c, weight, counts) * c.quality

    if len(candidates) == 0:
        raise ValueError("The candidates list can not be empty")

    # Find the candidate with the highest quality
    max_scoring_candidate = max(candidates, key = func_get_quality)
    max_score = score(max_scoring_candidate)
    # Use it to set the minimum score
    min_score = fraction * max_scoring_candidate.quality

    selected_candidates_count = 0
    logger.debug(f"var_cover: beam_width={beam_width} min_score={min_score}")
    
    # Select all subgroups that have score higher than the minimum
    while max_score >= min_score and selected_candidates_count < beam_width and len(candidates) > 0: # OMEGA(G, Sel) · φ(G) ≥ δ.
        logger.debug(f"SELECTED CANDIDATE N°{selected_candidates_count + 1}: {(max_score, max_scoring_candidate)}")
        # Add the highest scoring candidate to the beam
        beam.append(max_scoring_candidate)
        update_counts(max_scoring_candidate, counts)
        selected_candidates_count += 1
        # And remove it from the candidates list
        candidates.remove(max_scoring_candidate)
        # Update the highest scoring candidate
        (max_score, max_scoring_candidate) = max(((score(cand), cand) for cand in candidates), key=lambda c: c[0], default=(-1, None))

    return beam


def fixed_size_compression_beam_selection(candidates: List[Subgroup], beam: List[Subgroup]):
    raise NotImplementedError("Not implemented yet")


def var_size_compression_beam_selection(candidates: List[Subgroup]):
    raise NotImplementedError("Not implemented yet")


def apply_dominance_pruning(candidate: Subgroup, quality_func: FuncQuality, cover_func: FuncCover = None):
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
    """Insert the new candidate to keep the new list sorted descending and having at maximum the specified size"""
    
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

from . import selection_strategies, quality_measures, refinement_operators
def mine(data: DataFrame, column_types: Dict[str, ColumnType], descriptive_attributes: List[str], model_attributes: List[str], max_depth: int, k: int, j: int = math.inf, min_cov: int = 2, beam_width: int = 0, num_cut_points: Dict[str, int] = defaultdict(lambda: 5), 
    quality_measure: str = "", quality_parameters: Dict[str, Any] = {}, selection_strategy: str = "description", selection_params: Dict[str, Any] = {"min_diff_conditions": 2}, refinement_operator_name: str = "official", experience_id: str = "", save_intermediate_results: bool = True, post_selection_strategy: str = "", post_selection_params: Dict[str, Any] = {}, skip_phase2: bool = False, skip_phase3: bool = False) -> List[Subgroup]:
    """Mine and return mined subgroups

    Args:
        data (DataFrame): The dataset that mining operating will be made on
        column_types (Dict[str, ColumnType], optional): A mapping of every column in the dataset and their type.
        descriptive_attributes (List[str]): descriptive attributes used for mining (they obviously need to be a subset of all the attributes present in the dataset)
        model_attributes (List[str]): model attribute(s) depending on the quality measure that is selected
        max_depth (int): maximum number of conditions per subgroup pattern
        k (int): the number of subgroups to be returned as a result of the experiment
        j (int, optional): the maximum number of subgroups to be kept in memory at all time during the process. Defaults to math.inf.
        min_cov (int, optional): minimum coverage of selected subgroups. Defaults to 2.
        beam_width (int, optional): beam width. Defaults to 2.
        num_cut_points (Dict[str, int], optional): a map associating each numeric attribute with the number of cutpoints to use when discretizing that argument. Defaults to {}.
        quality_measure (str, optional): quality measure to be used. Defaults to "".
        quality_parameters (Dict[str, Any], optional): specific arguments required to instantiate the selection quality measure. Defaults to {}.
        selection_strategy (str, optional): selection strategy to be used. Defaults to "".
        selection_params (Dict[str, Any], optional): specific arguments required to instantiate the selected selection strategy. Defaults to {}.
        refinement_operator_name (str, optional): "name" of the refinement operator to be used. Defaults to "official".
        save_intermediate_results (bool, optional): whether or not to save intermediate results at each depth of the search phase (experimental for the moment, the purpose of this is to return an object that will allow to run only parts of the dssd algorithm step by step). Defaults to False.
        post_selection_strategy (str, optional): selection strategy to be used during post processing of the actual mining phase. Defaults to "".
        post_selection_params (Dict[str, Any], optional): specific arguments required to instantiate the selected post selection strategy. Defaults to {}.
        skip_phase2(bool): whether or not to skip phase 2. Defaults to False
        skip_phase3(bool): whether or not to skip phase 3. Defaults to False

    Returns:
        List[Candidate]:
    """

    # write a string version of all the arguments received to a config file, helpful to later remember what config yielded what result
    local_args = locals()
    post_selection_strategy = post_selection_strategy or selection_strategy
    post_selection_params = post_selection_params or selection_params
    local_args["column_types"] = {k: v.value for k,v in column_types.items()} # accessing the actual values of the enum objects
    function_args = "\n".join([f'{k}={v}' for k,v in local_args.items() if k not in ('self', 'data')])
    output_folder = f"outputs/{experience_id or f'{int(time.time())}-{selection_strategy}-max_depth={max_depth}'}"
    os.makedirs(output_folder, exist_ok=True)
    write_file(f"{output_folder}/dssd_params.conf", [function_args])


    logger = setup_logging(f"{output_folder}/stdout")
    dataset = Table(data, column_types)
    # a wrapper function that computes subgroups quality
    quality_func: FuncQuality = lambda c: q.compute_quality(q.dataset.loc[c.cover])
    # an "optimized" function to compute the cover of a subgroup, it is optimized cause it uses its parent cover as a base and only checks last added condition
    cover_func_optimized: FuncCover = lambda c: subgroup(dataset.df.loc[c.parent.cover], c.description, True)
    # normal unoptimized wrapper function for computing the cover of a subgroup
    cover_func_non_optimized: FuncCover = lambda c: subgroup(dataset.df, c.description, False)

    # creating the quality measure, selection strategy and refinement operator based on their arguments
    q = quality_measures.create(quality_measure, dataset.df[model_attributes], extra_parameters=quality_parameters)
    selector = selection_strategies.create(selection_strategy, extra_parameters=selection_params)
    post_selector = selection_strategies.create(post_selection_strategy, extra_parameters=post_selection_params)
    refinement_operator = refinement_operators.create(refinement_operator_name, extra_parameters={
        "dataset": Table(dataset.df[descriptive_attributes], column_types=sub_dict(dataset.column_types, descriptive_attributes)),
        "num_cut_points": num_cut_points,
        "min_cov": min_cov, 
        "cover_func": cover_func_optimized,
        "quality_func": quality_func
    })
    logger.info(f"Phase 1: Mining {j} subgroups each having at most {max_depth} conditions each")

    # initialize a beam with a base candidate subgroup, containing all elements of the initial dataset
    beam = [Subgroup(Description([]), 0, cover=dataset.df.index, parent=None)]
    result: List[Subgroup] = []
    start_time = time.time()
    depth = 1
    while depth <= max_depth:
        candidates: List[Subgroup] = []
        logger.info(f"Generating depth {depth} candidates")
        for cand in beam:
            # all valid candidates that extend current candidate's pattern with one condition
            candidates += refinement_operator.refine_candidate(cand, [])

        logger.info(f"depth={depth} : generated {len(candidates)} candidates")
        logger.info(f"depth={depth} : {min_max_avg_quality_string(candidates, ' ')}")
        for cand in candidates:
            update_topk(result, cand, j)
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-generated-candidates.csv", [to_csv(candidates)])

        beam = selector.select(candidates, beam_width=beam_width, beam = [])
        logger.info(f"depth={depth} : selected {len(beam)} candidates\n")
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-selected-candidates.csv", [to_csv(beam)])
        depth += 1

    # prune each candidate's pattern to only keep the actually useful conditions
    if save_intermediate_results: write_file(f"{output_folder}/stats1-after-mining-and-before-pruning.csv", [to_csv(result)])
    if not skip_phase2:
        logger.info(f"Phase 2: Dominance pruning & deduplication")
        logger.info(f"Dominance pruning {len(result)} candidates...")
        for cand in result:
            apply_dominance_pruning(cand, quality_func, cover_func_non_optimized)

        # remove duplicates that may have occured due to the pruning process
        logger.info(f"Removing duplicates...")
        result = remove_duplicates(result)
        logger.info(f"{len(result)} remaining after candidates deduplication...")
        if save_intermediate_results: write_file(f"{output_folder}/stats2-after-duplicates-removal.csv", [to_csv(result)])

    # final candidates selection / post selection phase
    if not skip_phase3:
        logger.info(f"Phase 3: Post selecting {k} candidates out of {len(result)}...")
        result = post_selector.select(result, beam_width=k, beam=[])
        sort_subgroups(result)

    logger.info(f"Total time taken = {time.time() - start_time} seconds\n")
    logger.info(f"Final selection\n{len(result)} candidate(s)\n{min_max_avg_quality_string(result, ' ')}")
    logger.info(f"Results stored in the folder: {os.getcwd()}/{output_folder}")
    write_file(f"{output_folder}/stats3-final-results.csv", [to_csv(result)])
    return result
