from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from .subgroup import Subgroup
from . import dssd


class SelectionStrategy(ABC):
    """
    A class representing a selection strategy used to select subgroups from a pool.
    This is a class instead of just a function to allow holding additional values used during the selection
    """

    @abstractmethod
    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
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


class FixedDescriptionBasedSelectionStrategy(SelectionStrategy):
    """
    Fixed description selection strategy\n
    Explanation:
        This strategy greedily selects sub-groups by comparing each candidate to the subgroups
        already selected. If there is a selected subgroup that has equal quality and the same
        conditions except for one, the candidate is skipped.

    Parameters
    ----------
    min_diff_conditions: int, default=2
        The number of conditions that have to differ for same quality candidate to be 
        included in the selection

    References
    ----------
        [1] Page 222
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    def __init__(self, min_diff_conditions: int = 2) -> None:
        super().__init__()
        self.min_diff_conditions = min_diff_conditions

    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        return dssd.fixed_size_description_selection(candidates=cands, beam=beam, beam_width=beam_width, min_diff_conditions=self.min_diff_conditions)


class VarDescriptionBasedSelectionStrategy(SelectionStrategy):
    """
    Variable size description selection strategy\n
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

    def current_depth(self, cands: List[Subgroup]) -> int:
        return len(cands[0].description.conditions)

    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        return dssd.var_size_description_selection(candidates=cands, beam=beam, beam_width=beam_width, c=self.max_attribute_occ, l=self.current_depth(cands))


class VarDescriptionBasedPostSelectionStrategy(VarDescriptionBasedSelectionStrategy):
    def current_depth(self, cands: List[Subgroup]) -> int:
        return max(len(cand.description.conditions) for cand in cands)


class FixedCoverBasedSelectionStrategy(SelectionStrategy):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        return dssd.fixed_size_cover_selection(candidates=cands, beam=beam, beam_width=beam_width, weight=self.weight)


class VarCoverBasedSelectionStrategy(FixedCoverBasedSelectionStrategy):
    def __init__(self, weight: float, fraction: float) -> None:
        super().__init__(weight)
        self.fraction = fraction

    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        return dssd.var_size_cover_selection(candidates=cands, beam=beam, beam_width=beam_width, weight=self.weight, fraction=self.fraction)


_builders: Dict[str, Callable[..., SelectionStrategy]] = None

if _builders is None: 
    _builders = {
        "description": FixedDescriptionBasedSelectionStrategy,
        "description-var": VarDescriptionBasedSelectionStrategy,
        "description-var-post": VarDescriptionBasedPostSelectionStrategy,
        "cover": FixedCoverBasedSelectionStrategy,
        "cover-var": VarCoverBasedSelectionStrategy,
        "compression": None
    }

def register(selection_strategy: str, builder: Callable[..., SelectionStrategy]):
    _builders[selection_strategy] = builder

def create(selection_strategy: str, extra_parameters: Dict[str, Any]) -> SelectionStrategy:
    builder = _builders.get(selection_strategy)
    if not builder:
            raise ValueError(f"!!! UNSUPPORTED SELECTION STRATEGY ({selection_strategy}) !!!")
    return builder(**extra_parameters)
