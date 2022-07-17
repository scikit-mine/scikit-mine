from ..cond import Cond

def test_cond_creation():
    cond = Cond("a", "<=", 700)
    assert cond.attribute == "a"
    assert cond.op == "<="
    assert cond.val == 700

def test_cond_eq():
    cond1 = Cond("weight", "<=", 700)
    cond2 = Cond("weight", "<=", 700)
    cond3 = Cond("", "", 0.)
    assert cond1 == cond2 
    assert cond1 != cond3
    assert cond2 != cond3

def test_cond_to_string():
    cond = Cond("a", "<", 5)
    assert str(cond) == "(a < 5)"
    assert repr(cond) == "(a < 5)"

def test_cond_hash():
    cond1 = Cond("a", ">", 6)
    cond2 = Cond("a", ">", 6)
    assert hash(cond1) == hash(cond2)

    conds_list = [cond1, cond2]

    assert cond1 == cond2
    conds_set = set(conds_list)

    assert cond1 in conds_set
    assert cond2 in conds_set
    assert len(conds_set) == 1