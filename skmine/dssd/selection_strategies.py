from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from typing import DefaultDict, Dict, List
from .custom_types import FuncQuality
from .utils import diff_items_count, sort_subgroups, func_get_quality, dummy_logger
from .subgroup import Subgroup


class SelectionStrategy(ABC):
    """
    A class representing a selection strategy used to select subgroups from a pool.
    This is a class instead of just a function to allow holding additional values used during the selection
    """

    @abstractmethod
    def select(self, cands: List[Subgroup], beam_width: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
        """
        Select a number of subgroups from the specified list into the received beam list

        Parameters
        ----------
        cands: List[Subgroup]
            The list to select from
        beam_width: int
            The number of subgroups to be selected
        beam: List[Subgroup]
            The destination of the selection

        Returns
        -------
        List[Subgroup]: Return the beam object received for convenience purposes
        """
        pass


def _fixed_size_description_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, min_diff_conditions: int = 2, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
    def is_candidate_diverse(candidate: Subgroup, beam: List[Subgroup], min_diff_conditions: int) -> bool:
        return all(c.quality != candidate.quality or diff_items_count(c.description.conditions, candidate.description.conditions) >= min_diff_conditions for c in beam)

    logger.debug(f"desc: beam_width={beam_width}")
    selected_candidates_count = 0
    candidate_index = 0
    # Make sure candidates are ordered in quality descending order
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

class Desc(SelectionStrategy):
    """
    Fixed description selection strategy\n
    Explanation:
        This strategy greedily selects sub-groups by comparing each candidate to the subgroups
        already selected. If there is a selected subgroup that has equal quality and the same
        conditions except for one, the candidate is skipped.

    Parameters
    ----------
    min_diff_conditions: int, default=2
        The number of conditions that have to differ for same quality candidate to be included in the selection

    References
    ----------
    [1] Page 222
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    def __init__(self, min_diff_conditions: int = 2) -> None:
        super().__init__()
        self.min_diff_conditions = min_diff_conditions

    def __str__(self) -> str:
        return f"desc[min_diff_conditions={self.min_diff_conditions}]"

    def select(self, cands: List[Subgroup], beam_width: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
        return _fixed_size_description_selection(candidates=cands, beam=[], beam_width=beam_width, min_diff_conditions=self.min_diff_conditions, logger=logger)



def _var_size_description_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, c: int, l: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
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

class VarDescFast(SelectionStrategy):
    """
    Variable size description selection strategy[fast variant better suited for phase 1 of the dssd algorithm]\n
    Explanation:
        An alternative way to achieve diversity is to allow each description attribute to occur 
        only c times in a condition in a subgroup set. Because the number of occurrences of 
        an attribute depends on the number of conditions per description, each attribute is allowed 
        to occur c * l times, where l is the (maximum) length of the  descriptions in the candidate set. 
        The beam width now depends on the number of description attributes |D|, c and l.
        This effectively results in a (more or less) static beam width per experiment

    Parameters
    ----------
    max_attribute_occ: int, default=2
        The maximum occurence of an attribute or c as per the exaplanation

    References
    ----------
    [1] Page 222
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    def __init__(self, max_attribute_occ: int = 2) -> None:
        super().__init__()
        self.max_attribute_occ = max_attribute_occ

    def __str__(self) -> str:
        return f"var_desc[max_attribute_occ={self.max_attribute_occ}]"

    def _current_depth(self, cands: List[Subgroup]) -> int:
        """
        Return the depth (or l) to be used for a set of candidates among.
        This version makes the assumption that all candidates in the cands
        list have the same description length.
        """
        return len(cands[0].description.conditions)

    def select(self, cands: List[Subgroup], beam_width: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
        return _var_size_description_selection(candidates=cands, beam=[], beam_width=beam_width, c=self.max_attribute_occ, l=self._current_depth(cands), logger=logger)


class VarDescStandard(VarDescFast):
    """
    Variable size description selection strategy[standard version with no optimization and is better suited for phase 3 of the dssd algorithm]

    See `VarDescriptionBasedSelectionStrategy` for full documentation on this strategy
    """
    def _current_depth(self, cands: List[Subgroup]) -> int:
        """
        Return the depth (or l) to be used for a set of candidates among and makes no 
        assumtion of the depth of the various candidates
        """
        return max(len(cand.description.conditions) for cand in cands)


def multiplicative_weighted_covering_score_smart(cand: Subgroup, counts: DefaultDict[int, int], weight: float) -> float:
    """
    Compute and return the weighted covering score for the current candidate based on 
    how often its cover overlaps with already selected candidates

    Requirement
    -----------
    len(cand.cover) > 0

    Parameters
    ----------
    cand: Subgroup
        The candidate to rank
    counts: DefaultDict[int, int]
        The counts of every transaction(index in the original entire dataset) 
        from already selected candidates
    weight: float
        The initial weight of every transaction

    Returns
    -------
    float

    References
    ----------
    [1] Page 222
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
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
    """Increase by 1 the counts of every transaction in the specified candidate's cover"""
    for t in cand.cover:
        counts[t] += 1
        
    
def _fixed_size_cover_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, weight: float, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
    counts = defaultdict(int, {})
    score: FuncQuality = lambda c: multiplicative_weighted_covering_score_smart(c, counts, weight) * c.quality

    # In case there are less candidates than the beam width just retrun the candidates list
    if len(candidates) <= beam_width:
        beam.extend(candidates)
        return beam

    logger.debug(f"cover: beam_width={beam_width}")

    for i in range(beam_width):
        logger.debug(f"SELECTED CANDIDATE {i + 1}")

        # Favoring subgroup with shorter when having equal quality, the shorter the description the better
        max_scoring_candidate = max(candidates, key = lambda cand: (score(cand), -len(cand.description)))

        # Select the the candidate with the highest score for beam_width times
        candidates.remove(max_scoring_candidate)
        beam.append(max_scoring_candidate)
        update_counts(max_scoring_candidate, counts)

    return beam

class Cover(SelectionStrategy):
    """
    Fixed size cover based selection strategy\n
    Explanation:
        A score based on multiplicative weighted covering (Lavraˇc et al 2004) is used to weigh 
        the quality of each subgroup, aiming to minimise the overlap between the selected subgroups.
        The less often tuples in subgroup G are already covered by subgroups in the selection, 
        the larger the score. If the cover contains only previously uncovered tuples, wscore(G, Sel) = 1.
        In k iterations, k subgroups are selected. In each iteration, the subgroup that maximises 
        weighted_score(G, Sel) · quality(subgroup) is selected.

    Parameters
    ----------
    weight: float, default=0.9
        The weight to use for computing covering score. Should be in ]0, 1[
        The lower this weight, the less likely there will be an overalap in the selection

    References
    ----------
    [1] Page 222
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    
    def __init__(self, weight: float = 0.9) -> None:
        super().__init__()
        self.weight = weight

    def __str__(self) -> str:
        return f"cover[weight={self.weight}]"

    def select(self, cands: List[Subgroup], beam_width: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
        return _fixed_size_cover_selection(candidates=cands, beam=[], beam_width=beam_width, weight=self.weight, logger=logger)


def _var_size_cover_selection(candidates: List[Subgroup], beam: List[Subgroup], beam_width: int, weight: float, fraction: float, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
    counts = defaultdict(int, {})
    score: FuncQuality = lambda c: multiplicative_weighted_covering_score_smart(c, counts,  weight) * c.quality

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
    while max_score >= min_score and selected_candidates_count < beam_width and len(candidates) > 0:
        logger.debug(f"SELECTED CANDIDATE N°{selected_candidates_count + 1}: {(max_score, max_scoring_candidate)}")
        # Add the highest scoring candidate to the beam
        beam.append(max_scoring_candidate)
        update_counts(max_scoring_candidate, counts)
        selected_candidates_count += 1
        # And remove it from the candidates list
        candidates.remove(max_scoring_candidate)
        # Update the highest scoring candidate with the shortest description, this is quite useful when subgroups in the pool don't have the same description length
        (max_score, max_scoring_candidate) = max(((score(cand), cand) for cand in candidates), key=lambda c: (c[0], -len(c[1].description)), default=(-1, None))
        

    return beam


class VarCover(Cover):
    """
    Variable size cover based selection strategy\n
    Explanation:
        This selection procedure is equivalent to the fixed-size version, except for the stopping criterion. 
        Subgroups are iteratively selected until no candidate subgroup meets the minimum score specified by param-
        eter f. The minimum score is defined as fraction x quality of the top-ranking candidate. 
        Selection stops when there is no subgroup which weighted covering score is greater or equal to the min score

    Parameters
    ----------
    weight: float, default=0.9
        The weight to use for computing covering score. Should be in ]0, 1[
        The lower this weight, the less likely there will be an overalap in the selection
    fraction: float, default=0.5
        The fraction to use for computing the min_score. Should be in [0, 1[

    References
    ----------
    [1] Page 223
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    
    def __init__(self, weight: float = 0.9, fraction: float = 0.5) -> None:
        super().__init__(weight)
        self.fraction = fraction

    def __str__(self) -> str:
        return f"var_cover[weight={self.weight}-fraction={self.fraction}]"

    def select(self, cands: List[Subgroup], beam_width: int, logger: logging.Logger = dummy_logger) -> List[Subgroup]:
        return _var_size_cover_selection(candidates=cands, beam=[], beam_width=beam_width, weight=self.weight, fraction=self.fraction, logger=logger)
