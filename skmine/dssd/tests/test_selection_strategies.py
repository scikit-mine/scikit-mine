from collections import defaultdict
import pandas
import pytest

from ..cond import Cond
from ..subgroup import Subgroup
from ..description import Description
from ..selection_strategies import (
    _fixed_size_cover_selection, 
    _var_size_cover_selection, 
    _var_size_description_selection, 
    multiplicative_weighted_covering_score_smart,
    FixedCoverBasedSelectionStrategy, 
    FixedDescriptionBasedSelectionStrategy, 
    VarCoverBasedSelectionStrategy, 
    VarDescriptionBasedStandardSelectionStrategy, 
    VarDescriptionBasedFastSelectionStrategy
)

def test_fixed_size_description_selection():
    s = FixedDescriptionBasedSelectionStrategy(min_diff_conditions = 2)

    empty_cand = Subgroup(Description([]), 0.0)
    beam = [Subgroup(Description([]), 0.0)]
    beam_width = 3

    # testing some edges cases
    assert s.select([], beam=[], beam_width=beam_width) == []
    assert s.select([], beam=beam, beam_width=beam_width) == [empty_cand]
    assert s.select([empty_cand], beam=[], beam_width=beam_width) == [empty_cand]
    assert s.select([empty_cand], beam=[empty_cand], beam_width=beam_width) == [empty_cand]
    assert s.select([empty_cand, empty_cand], beam=[empty_cand], beam_width=beam_width) == [empty_cand]
    
    # regular cases
    cand1 = Subgroup(Description([Cond("a", "==", 0)]), 0.0)
    assert s.select([cand1, cand1], beam=[empty_cand], beam_width=beam_width) == [empty_cand]

    # candidates with the same quality and not enough different patterns is not selected
    cand1 = Subgroup(Description([Cond("a", ">", 0)]), 0.4)
    cand2 = Subgroup(Description([Cond("a", ">", 0), Cond("a", ">", 6)]), 0.4)
    # cand2 has only one condition different from cand1 so it is not selected
    assert s.select([cand1, cand2], beam=[empty_cand], beam_width=beam_width) == [empty_cand, cand1]

    cand3 = Subgroup(Description([Cond("a", ">", 0), Cond("a", ">", 6), Cond("a", ">", 7)]), 0.4)
    # showing that although it may look logical for cand3 to not be selected too, it will be as it has two conditions that no selected candidate has
    # dominance pruning will eliminate these candidates
    assert s.select([cand1, cand2, cand3], beam=[empty_cand], beam_width=beam_width) == [empty_cand, cand1, cand3]

    # ensuring that the at the number of selected candidates is at most the beam_width
    cand1 = Subgroup(Description([Cond("a", "!=", "three"), Cond("a", "!=", "four"), Cond("a", "!=", "one")]), 0.2)
    cand2 = Subgroup(Description([Cond("a", "==", "two")]), 0.2)
    cand3 = Subgroup(Description([Cond("a", "==", "three")]), 0.3)
    cand4 = Subgroup(Description([Cond("a", "==", "four")]), 0.4)
    # cand5 = Candidate(Description([Cond("a", "==", "five")]), 0.5)
    # cand6 = Candidate(Description([Cond("a", "==", "six")]), 0.6)
    res = s.select([cand3, cand4, cand1, cand2], beam=[empty_cand], beam_width=beam_width)
    print(res)
    # Notes: One could argue here that cand2 should have been selected instead of cand1
    # but because of the short beam width and as they have same quality and comparison between candidates is only implemented
    # using the quality, the selected one here really just based on how the sorting algorithm is implemented
    # so if the sorting algorithm switches places of equal items then the result might be different
    # We could also include the length of the candidate description while comparing candidates but that
    # could creates worst issues as a long pattern can have an equal quality as a short one but still 
    # targets a different subgroup
    assert res == [empty_cand, cand4, cand3, cand1]

    # as shown in this test, we can see that when the beam is wide enough to accomodate cand1 and cand2, they are both selected
    res = s.select([cand3, cand4, cand1, cand2], beam=[empty_cand], beam_width=beam_width + 1)
    print(res)
    assert res == [empty_cand, cand4, cand3, cand1, cand2]


def test_var_size_description_selection():
    s = VarDescriptionBasedFastSelectionStrategy(max_attribute_occ = 1)

    # c is the number of times an attribute is allowed to appear in a description
    # l is the maximum number of conditions that a single candidate can have
    c = 0
    l = 0

    # the qualities here are completely fictionnal as only the logic of the algorithm is taken into account here
    cand1 = Subgroup(Description([Cond("a", "<", 0)]), 5)
    cand2 = Subgroup(Description([Cond("a", ">", 6)]), 4)
    cand3 = Subgroup(Description([Cond("b", "<", 6)]), 3)
    cand4 = Subgroup(Description([Cond("b", ">", 6)]), 4)
    
    cands1 = [cand1,cand2,cand3,cand4]

    # testing edge case beam_width = 0, c or l = 0
    assert _var_size_description_selection(cands1, [], 0, 1, 1) == []
    assert _var_size_description_selection(cands1, [], 5, 1, 0) == []
    # the result is only the highest quality candidate as the beam width is 1
    assert s.select(cands1, beam=[], beam_width=1)  == [cand1]

    # as each attribute is allowed to be used 1 * 1 time(s) in the resulting beam
    # only the highest quality candidates are selected and the others contributing
    # into overusing the attributes are discarded even though the beam_width is not 
    # yet reached 
    assert s.select(cands1, beam=[], beam_width=3)  == [cand1, cand4]

    cand1 = Subgroup(Description([Cond("a", "<", 0), Cond("b", "<", 6)]), 5)
    cand2 = Subgroup(Description([Cond("b", ">", 6), Cond("b", ">", 7)]), 4)
    cand3 = Subgroup(Description([Cond("a", ">", 6), Cond("b", "<", 6)]), 4)
    cand4 = Subgroup(Description([Cond("a", "<", 6), Cond("a", "<", 2)]), 3)
    cands2 =  [cand1,cand2,cand3,cand4]

    # cand1 is selected cause it has the highest quality and the beam is initially selected
    # cand2 it selected cause it has the second highest quality and cause of the initial order, but after its selection, it causes the usage of the b attribute to overflow
    # cand3 is not selected cause the max usage of attribute b is already reached
    # cand4 is then selected as the usage of a is not yet reached
    # although the beam_width is not yet reached, the alogorithm stops as there is no candidate left to examine 
    assert s.select(cands2, beam=[], beam_width=4)  == [cand1, cand2, cand4]


def test_multiplicative_weighted_covering_score():
    empty_cand = Subgroup(Description([]), cover=pandas.Index([]))

    # ensure only non empty candidates can be given as argument
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score_smart(empty_cand, {}, 0.9)

    # ensure a non empty candidate is given as an argument
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score_smart(empty_cand, {}, 0)

    # ensure invalid weigh raises an exception
    with pytest.raises(ValueError):
        multiplicative_weighted_covering_score_smart(Subgroup(Description([]), cover=pandas.Index([0])), {}, 1.5)

    # non empty cover with empty selection
    assert multiplicative_weighted_covering_score_smart(Subgroup(Description([]), 0.0, cover=pandas.Index([0, 2])), defaultdict(int), .9) == 1

    # watch score decrease as the selection already contains transactions covered by the candidate
    cand1 = Subgroup(Description([]), cover=pandas.Index([0]))
    assert multiplicative_weighted_covering_score_smart(cand1, {0: 1}, 0.9) == .9
    assert multiplicative_weighted_covering_score_smart(cand1, {0: 2}, 0.9) == .9 ** 2

    # candidate with a cover size different that those already in the selection
    # candidate size is taken into account while computing the score 
    cand2 = Subgroup(Description([]), cover=pandas.Index([0, 5, 6]))
    assert multiplicative_weighted_covering_score_smart(cand2, defaultdict(int, {0: 2}), 0.9) == (.9 ** 2 + 1 + 1) / len(cand2.cover)


def test_fixed_size_cover_selection():
    s = FixedCoverBasedSelectionStrategy(weight = .5)

    # descriptions here are not used so we could have left them empty
    cand1 = Subgroup(Description(), 7.5, pandas.Index([1, 2, 3, 4, 5, 6]))
    cand2 = Subgroup(Description([Cond("a", "<", 0)]), 7.5, pandas.Index([1, 2, 3, 4, 5, 7]))
    cand3 = Subgroup(Description(),  5, pandas.Index([1, 2, 5]))
    cand4 = Subgroup(Description(),  4.4, pandas.Index([8, 9, 10]))

    cands = [cand2, cand3, cand1, cand4]
    
    assert s.select(cands, beam=[], beam_width=4) == [cand2, cand4, cand1, cand3]

    beam_width = 2
    res = s.select(cands, beam=[], beam_width=beam_width)
    
    # ensure that the result is equal to the beam width provided that the initial list of candidates if large enough
    assert len(cands) < beam_width or len(res) == beam_width

    # ensure that the first selected candidate has the highest quality
    assert res[0].quality == max(c.quality for c in cands) == cand2.quality

    # here the first selected candidate is either cand1 or cand2
    # because they both have highest quality and very similar subgroups 
    # so the other one will be discarded 
    assert res[0].cover.equals(cand1.cover) or res[0].cover.equals(cand2.cover)


    # at the second pass, this the scores are as follows:
    # score(cand1 or cand2 that was not selected) = 1/6 * (0.5 * 5 + 1) * 7.5 = 4.375
    # score(cand3) = 1/3 * (0.5 * 3) * 5 = 2.5
    # score(cand4) = 1/3 * (1 * 3) * 4.4 = 4.4
    # cand4 has the highest score so it is selected and as the beam width is reached the procedure stops
    assert multiplicative_weighted_covering_score_smart(cand3, defaultdict(int, {k: 1 for k in cand1.cover}), s.weight) * cand3.quality == 2.5
    assert multiplicative_weighted_covering_score_smart(cand4, defaultdict(int, {k: 1 for k in cand1.cover}), s.weight) * cand4.quality == 4.4
    assert res[1] == cand4


def test_var_size_cover_selection():
    cand1 = Subgroup(Description(), 7.5, pandas.Index([1, 2, 3, 4, 5, 6]))
    cand2 = Subgroup(Description([Cond("a", "<", 0)]), 7.5, pandas.Index([1, 2, 3, 4, 5, 7]))
    cand3 = Subgroup(Description(),  5, pandas.Index([1, 2, 5]))
    cand4 = Subgroup(Description(),  4.4, pandas.Index([8, 9, 10]))

    cands = [cand2, cand3, cand1, cand4]
    weight = 0.5
    beam_width = 3
    # fraction explanation at the last line of this function
    fraction = 4.375 / 7.5 + 0.001
    s = VarCoverBasedSelectionStrategy(weight = .5, fraction= fraction)

    # empty candidates list
    with pytest.raises(ValueError):
        _var_size_cover_selection([], [], 4, weight, 0)

    # fixed size version is the same as variable size version but with 0 as the fraction argument
    assert _var_size_cover_selection(cands.copy(), [], 4, weight, 0) == _fixed_size_cover_selection(cands.copy(), [], 4, weight)

    # ensure that the result is equal to the beam width provided that the initial list of candidates if large enough
    res = s.select(cands.copy(), beam=[], beam_width=beam_width)
    assert len(res) <= beam_width
    assert len(res) == 2

    # ensure that the first selected candidate has the highest quality
    assert res[0].quality == max(c.quality for c in cands) == cand2.quality

    # ensure the first selected candidate is either cand1 or cand2 as they both have highest quality
    assert res[0].cover.equals(cand1.cover) or res[0].cover.equals(cand2.cover)

    # the minimum score for this experiment is: (4.375 / 7.5 + 0.001) * 7.5 = 4.3825
    # at the second pass, this the scores are as follows:
    # score(cand1 or cand2 that was not selected) = 1/6 * (0.5 * 5 + 1) * 7.5 = 4.375
    # score(cand3) = 1/3 * (0.5 * 3) * 5 = 2.5
    # score(cand4) = 1/3 * (1 * 3) * 4.4 = 4.4
    # cand4 has the highest score so it is selected
    assert res[1] == cand4

    # At the third pass, the scores are as follows:
    # score(cand1) = 1/6 * (0.5 * 5 + 1) * 7.5 = 4.375
    # score(cand3) = 1/3 * (0.5 * 3) * 5 = 2.5
    # As none of the two has a score higher than the minimum, the procedure stops here

