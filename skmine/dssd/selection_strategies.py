from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from .subgroup import Subgroup
from . import dssd


class SelectionStrategy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        pass


class FixedDescriptionBasedSelectionStrategy(SelectionStrategy):
    def __init__(self, min_diff_conditions: int = 2) -> None:
        super().__init__()
        self.min_diff_conditions = min_diff_conditions

    def select(self, cands: List[Subgroup], beam_width: int, beam: List[Subgroup] = []) -> List[Subgroup]:
        return dssd.fixed_size_description_selection(candidates=cands, beam=beam, beam_width=beam_width, min_diff_conditions=self.min_diff_conditions)


class VarDescriptionBasedSelectionStrategy(SelectionStrategy):
    def __init__(self, max_attribute_occ: int) -> None:
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
