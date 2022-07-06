from ..cond import Cond
from ..description import Description 


def test_description_default_creation(): 
    # ensure default arguments work as expected
    desc = Description()
    assert desc.conditions == []
    assert desc.op == "&"

    # ensure speicified arguments are taken into account
    conds = [Cond("a", "<", 2), Cond("b", ">", 5)]
    desc = Description(conds, "|")
    assert desc.conditions == conds
    assert desc.op == "|"


def test_description_contains():
    # testing all contains like method on description objects
    cond1 = Cond("a", "<", 2)
    same_cond1 = Cond("a", "<", 2)
    cond2 = Cond("b", "==", 5)
    absent_cond = Cond("4", "4", 0.)
    conds = [cond1, cond2]

    desc = Description(conds, "&")


    assert len(desc) == len(desc.conditions) == 2
    assert cond1 in desc.conditions and cond2 in desc.conditions
    assert cond1 in desc and cond2 in desc
    assert same_cond1 in desc
    assert absent_cond not in desc

    assert desc.is_attribute_used(cond1.attribute)
    assert desc.is_attribute_used(cond2.attribute)
    assert desc.is_attribute_used(same_cond1.attribute)
    assert not desc.is_attribute_used(absent_cond.attribute)

    assert desc.has_equal_condition_on_attr("b")
    assert not desc.has_equal_condition_on_attr("a")
    assert not desc.has_equal_condition_on_attr("absent")


def test_description_equality():
    assert Description() == Description()

    assert Description([Cond("b", ">", 5)]) == Description([Cond("b", ">", 5)])

    assert Description([Cond("b", ">", 5)]) != Description([Cond("a", ">", 5)])


def test_description_to_string():
    assert str(Description([Cond("b", ">", 5), Cond("b", "<", 10)])) is not None
