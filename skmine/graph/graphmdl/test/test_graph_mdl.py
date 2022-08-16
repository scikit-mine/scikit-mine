import networkx as nx
import pytest

import skmine.graph.graphmdl.utils as utils
from ..candidate import Candidate
from ..code_table import CodeTable
from ..code_table_row import CodeTableRow
from ..graph_mdl import GraphMDL
from ..label_codes import LabelCodes

pattern = nx.DiGraph()
pattern.add_node(1)
pattern.add_node(2)
pattern.add_edge(1, 2)

row = CodeTableRow(pattern)

""" Code Table Row"""


def test_pattern():
    assert len(row.pattern().nodes()) == 2
    assert len(row.pattern().edges()) == 1


def test_pattern_code():
    assert row.pattern_usage() is None


def test_set_pattern_code():
    row.set_pattern_usage(3.0)
    assert row.pattern_usage() == 3.0


def test_pattern_port_code():
    assert row.pattern_port_usage() is None


def test_embeddings():
    assert len(row.embeddings()) == 0


def test_set_embeddings():
    row.set_embeddings([0, 1, 2, 3])
    assert len(row.embeddings()) == 4
    assert row.embeddings()[0] == 0
    assert row.embeddings()[3] == 3


""" Label codes"""


def init_label_codes_test():
    res = dict()
    g = nx.DiGraph()
    g.add_nodes_from(range(1, 9))
    g.add_edge(2, 1, label='a')
    g.add_edge(4, 1, label='a')
    g.add_edge(6, 1, label='a')
    g.add_edge(6, 8, label='a')
    g.add_edge(8, 6, label='a')
    g.add_edge(1, 3, label='b')
    g.add_edge(1, 5, label='b')
    g.add_edge(1, 7, label='b')
    g.nodes[1]['label'] = 'y'
    g.nodes[2]['label'] = 'x'
    g.nodes[3]['label'] = 'z'
    g.nodes[4]['label'] = 'x'
    g.nodes[5]['label'] = 'z'
    g.nodes[6]['label'] = 'x'
    g.nodes[7]['label'] = 'z'
    g.nodes[8]['label'] = 'w', 'x'

    graph = g
    lc = LabelCodes(graph)
    res['st'] = lc
    res['graph'] = graph
    return res


def test_total_label():
    st = init_label_codes_test()['st']
    assert st.total_label() == 17.0


def test_vertex_st():
    st = init_label_codes_test()['st']
    assert pytest.approx(st.vertex_lc()['y'], rel=1e-01) == 4.09
    assert pytest.approx(st.vertex_lc()['x'], rel=1e-01) == 2.09
    assert pytest.approx(st.vertex_lc()['z'], rel=1e-01) == 2.50
    assert pytest.approx(st.vertex_lc()['w'], rel=1e-01) == 4.09


def test_edges_st():
    st = init_label_codes_test()['st']
    assert pytest.approx(st.edges_lc()['a'], rel=1e-01) == 1.77
    assert pytest.approx(st.edges_lc()['b'], rel=1e-01) == 2.50


def test_encode():
    st = init_label_codes_test()['st']
    g1 = nx.DiGraph()
    g1.add_nodes_from(range(1, 3))
    g1.add_edge(1, 2, label='a')
    g1.nodes[1]['label'] = 'x'
    assert pytest.approx(st.encode(g1), rel=1e-01) == 21.44


def test_encode_singleton_vertex():
    st = init_label_codes_test()['st']
    assert pytest.approx(st.encode_singleton_vertex('x'), rel=1e-01) == 12.67


def test_encode_singleton_edge():
    st = init_label_codes_test()['st']
    assert pytest.approx(st.encode_singleton_edge('a'), rel=1e-01) == 14.35


""" Utils test"""


def init_graph_to_utils_test():
    res = dict()
    g = nx.DiGraph()
    g.add_nodes_from(range(1, 9))
    g.add_edge(2, 1, label='a')
    g.add_edge(4, 1, label='a')
    g.add_edge(6, 1, label='a')
    g.add_edge(6, 8, label='a')
    g.add_edge(8, 6, label='a')
    g.add_edge(1, 3, label='b')
    g.add_edge(1, 5, label='b')
    g.add_edge(1, 7, label='b')
    g.nodes[1]['label'] = 'y'
    g.nodes[2]['label'] = 'x'
    g.nodes[3]['label'] = 'z'
    g.nodes[4]['label'] = 'x'
    g.nodes[5]['label'] = 'z'
    g.nodes[6]['label'] = 'x'
    g.nodes[7]['label'] = 'z'
    g.nodes[8]['label'] = 'w', 'x'
    label_codes = LabelCodes(g)
    res['graph'] = g
    res['st'] = label_codes
    return res


def init_multi_graph():
    graph = nx.MultiDiGraph()
    graph.add_node(1, label="Book")
    graph.add_node(2, label="Book")
    graph.add_node(3, label="Book")
    graph.add_node(4)
    graph.nodes[4]['label'] = 'xsd:string', 'Value:Alice'
    graph.add_node(5, label="Person")
    graph.add_node(6, label="Person")
    graph.add_node(7)
    graph.nodes[7]['label'] = 'xsd:string', 'Value:Bob'
    graph.add_node(8, label="City")
    graph.add_node(9, label="City")
    graph.add_node(10, label="Monument")
    graph.add_node(11, label="Monument")
    graph.add_node(12)
    graph.add_node(13)
    graph.nodes[12]['label'] = 'xsd:integer', 'Value:123'
    graph.nodes[13]['label'] = 'xsd:integer', 'Value:123'
    graph.add_edge(1, 5, label='author')
    graph.add_edge(2, 5, label='author')
    graph.add_edge(3, 5, label='author')
    graph.add_edge(3, 6, label='author')
    graph.add_edge(5, 4, label='name')
    graph.add_edge(5, 8, label='born_in')
    graph.add_edge(5, 8, label='died_in')
    graph.add_edge(6, 7, label='name')
    graph.add_edge(6, 9, label='born_in')
    graph.add_edge(6, 9, label='died_in')
    graph.add_edge(10, 8, label='is_located')
    graph.add_edge(11, 8, label='is_located')
    graph.add_edge(10, 12, label='height')
    graph.add_edge(10, 11, label='near')
    graph.add_edge(11, 10, label='near')
    graph.add_edge(11, 13, label='height')
    label_codes = LabelCodes(graph)
    ct = CodeTable(label_codes, graph)
    ct.cover()

    p1 = nx.MultiDiGraph()
    p1.add_node(1)
    p1.add_node(2, label='Monument')
    p1.add_node(3, label='Monument')
    p1.add_node(4)
    p1.add_node(5)
    p1.nodes[4]['label'] = 'xsd:integer', 'Value:123'
    p1.nodes[5]['label'] = 'xsd:integer', 'Value:123'
    p1.add_edge(2, 1, label='is_located')
    p1.add_edge(3, 1, label='is_located')
    p1.add_edge(2, 4, label='height')
    p1.add_edge(2, 3, label='near')
    p1.add_edge(3, 2, label='near')
    p1.add_edge(3, 5, label='height')
    row1 = CodeTableRow(p1)

    p2 = nx.MultiDiGraph()
    p2.add_node(1)
    p2.add_node(2, label='City')
    p2.add_node(3, label='xsd:string')
    p2.add_edge(1, 3, label='name')
    p2.add_edge(1, 2, label='born_in')
    p2.add_edge(1, 2, label='died_in')
    row2 = CodeTableRow(p2)

    p3 = nx.MultiDiGraph()
    p3.add_node(1, label='Book')
    p3.add_node(2, label='Person')
    p3.add_edge(1, 2, label='author')
    row3 = CodeTableRow(p3)
    return {
        'graph': graph, 'label_codes': label_codes, 'ct': ct,
        'row1': row1, 'row2': row2, 'row3': row3}


def test_prequential_code():
    assert utils.prequential_code([0, 0]) == 0.0
    assert utils.prequential_code([1, 0]) == 1.0
    assert pytest.approx(utils.prequential_code([1, 1, 1, 1]), rel=1e-01) == 10.91
    assert pytest.approx(utils.prequential_code([2, 2, 4]), rel=1e-01) == 15.15
    assert pytest.approx(utils.prequential_code([10, 8, 0, 0, 0, 0, 0]), rel=1e-01) == 29.30
    assert pytest.approx(utils.prequential_code([100, 500, 4, 10, 0]), rel=1e-01) == 515.90


def test_count_edge_label():
    graph = init_graph_to_utils_test()['graph']
    assert len(utils.count_edge_label(graph).items()) == 2
    assert utils.count_edge_label(graph)['a'] == 5
    assert utils.count_edge_label(graph)['b'] == 3


def test_count_vertex_label():
    graph = init_graph_to_utils_test()['graph']
    assert len(utils.count_vertex_label(graph).items()) == 4
    assert utils.count_vertex_label(graph)['x'] == 4
    assert utils.count_vertex_label(graph)['y'] == 1
    assert utils.count_vertex_label(graph)['z'] == 3
    assert utils.count_vertex_label(graph)['w'] == 1


def test_get_total_label():
    graph = init_graph_to_utils_test()['graph']
    assert utils.get_total_label(graph) == 17


def test_binomial():
    with pytest.raises(ValueError):
        utils.binomial(2, 5)

    assert utils.binomial(2, 0) == 1
    assert utils.binomial(4, 3) == 4
    assert utils.binomial(4, 2) == 6


def test_universal_integer_encoding():
    with pytest.raises(ValueError):
        utils.universal_integer_encoding(0)

    assert utils.universal_integer_encoding(1) == 1


def test_universal_integer_encoding_with0():
    with pytest.raises(ValueError):
        utils.universal_integer_encoding_with0(-1)

        assert utils.universal_integer_encoding_with0(1) == 1


def test_utils_encode():
    standard_table = init_graph_to_utils_test()['st']
    graph = init_graph_to_utils_test()['graph']
    g1 = nx.DiGraph()
    g1.add_nodes_from(range(1, 3))
    g1.add_edge(1, 2, label='a')
    g1.nodes[1]['label'] = 'x'

    p5 = nx.DiGraph()
    p5.add_node(1, label='x')
    p5.add_node(2, label='y')
    p5.add_edge(1, 2, label='a')
    # print(utils.encode(p5, standard_table))

    assert pytest.approx(utils.encode(g1, standard_table), rel=1e-01) == 21.44
    assert pytest.approx(utils.encode(graph, standard_table), rel=1e-01) == 111.76


def test_encode_vertex_singleton():
    standard_table = init_graph_to_utils_test()['st']
    with pytest.raises(ValueError):
        utils.encode_vertex_singleton(standard_table, '')

    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'x'), rel=1e-01) == 12.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'y'), rel=1e-01) == 14.67
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'z'), rel=1e-01) == 13.09
    assert pytest.approx(utils.encode_vertex_singleton(standard_table, 'w'), rel=1e-01) == 14.67


def test_encode_edge_singleton():
    standard_table = init_graph_to_utils_test()['st']
    with pytest.raises(ValueError):
        utils.encode_edge_singleton(standard_table, '')

    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_edge_singleton(standard_table, 'b'), rel=1e-01) == 15.09


def test_encode_singleton():
    standard_table = init_graph_to_utils_test()['st']
    with pytest.raises(ValueError):
        utils.encode_singleton(standard_table, 0, 'a')

    assert pytest.approx(utils.encode_singleton(standard_table, 2, 'a'), rel=1e-01) == 14.35
    assert pytest.approx(utils.encode_singleton(standard_table, 1, 'x'), rel=1e-01) == 12.67

    # Test for graph


def init_ng_to_utils_test():
    res = dict()

    ng = nx.Graph()
    ng.add_nodes_from(range(1, 6))
    ng.add_edge(1, 2, label='e')
    ng.add_edge(2, 3, label='e')
    ng.add_edge(2, 4, label='e')
    ng.add_edge(5, 2, label='e')
    ng.nodes[1]['label'] = 'A'
    ng.nodes[2]['label'] = 'A'
    ng.nodes[3]['label'] = 'B'
    ng.nodes[4]['label'] = 'B'
    ng.nodes[5]['label'] = 'A'
    res['ng'] = ng
    pattern = nx.Graph()
    pattern.add_nodes_from(range(1, 3))
    pattern.add_edge(1, 2, label='e')
    pattern.nodes[1]['label'] = 'A'
    res['pattern'] = pattern
    return res


def test_get_embeddings():
    ng = init_ng_to_utils_test()['ng']
    pattern = init_ng_to_utils_test()['pattern']
    ng2 = nx.DiGraph()
    ng2.add_nodes_from(range(1, 6))
    ng2.add_edge(1, 2, label='e')
    ng2.add_edge(2, 3, label='e')
    ng2.add_edge(2, 4, label='e')
    ng2.add_edge(5, 2, label='e')
    ng2.nodes[1]['label'] = 'A'
    ng2.nodes[2]['label'] = 'A'
    ng2.nodes[3]['label'] = 'B'
    ng2.nodes[4]['label'] = 'B'
    ng2.nodes[5]['label'] = 'A'

    ngp = nx.DiGraph()
    ngp.add_nodes_from(range(1, 3))
    ngp.add_edge(1, 2, label='e')
    ngp.nodes[1]['label'] = 'A'
    # first test for digraph
    assert len(utils.get_embeddings(ngp, ng2)) == 4
    assert utils.get_embeddings(ngp, ng2)[0][1] == 1
    assert utils.get_embeddings(ngp, ng2)[0][2] == 2

    embed = utils.get_embeddings(pattern, ng)
    assert len(utils.get_embeddings(pattern, ng)) == 6
    assert utils.get_embeddings(pattern, ng)[5][5] == 1
    assert utils.get_embeddings(pattern, ng)[5][2] == 2

    test1 = nx.Graph()
    ptest1 = nx.Graph()
    test1.add_node(1)
    test1.nodes[1]['label'] = 'w', 'x', 'y'
    ptest1.add_node(1)
    ptest1.nodes[1]['label'] = 'w', 'x'

    assert len(utils.get_embeddings(ptest1, test1)) != 0
    assert len(utils.get_embeddings(test1, ptest1)) == 0

    test2 = nx.Graph()
    test2.add_node(1)
    ptest2 = nx.Graph()
    ptest2.add_node(1, label='x')
    assert len(utils.get_embeddings(ptest2, test2)) == 0
    assert len(utils.get_embeddings(test2, ptest2)) != 0


def test_is_vertex_singleton():
    g1 = nx.DiGraph()
    g1.add_node(1, label='a')
    assert utils.is_vertex_singleton(g1) is True

    g2 = nx.Graph()
    g2.add_node(1, label='a')
    assert utils.is_vertex_singleton(g2) is True

    g3 = nx.Graph()
    g3.add_nodes_from(range(1, 3))
    assert utils.is_vertex_singleton(g3) is False

    g4 = nx.DiGraph()
    g4.add_node(1)
    g4.nodes[1]['label'] = 'a', 'b'
    assert utils.is_vertex_singleton(g4) is False


def test_is_edge_singleton():
    g1 = nx.DiGraph()
    g1.add_node(1, label='a')
    assert utils.is_edge_singleton(g1) is False

    g2 = nx.Graph()
    g2.add_node(1)
    g2.add_node(2, label='a')
    assert utils.is_edge_singleton(g2) is False

    g4 = nx.Graph()
    g4.add_node(1)
    g4.add_node(2)
    g4.add_edge(1, 2)
    g4[1][2]['label'] = 'a', 'b'
    print(bool(utils.count_edge_label(g4)))
    assert utils.is_edge_singleton(g4) is False

    g5 = nx.DiGraph()
    g5.add_node(1)
    g5.add_node(2)
    g5.add_edge(1, 2)
    g5[1][2]['label'] = 'a'
    assert utils.is_edge_singleton(g5) is True


def test_get_support():
    ng = init_ng_to_utils_test()['ng']
    pattern = init_ng_to_utils_test()['pattern']
    graph = init_graph_to_utils_test()['graph']
    pattern1 = nx.DiGraph()
    pattern1.add_nodes_from(range(1, 4))
    pattern1.nodes[1]['label'] = 'x'
    pattern1.nodes[2]['label'] = 'y'
    pattern1.nodes[3]['label'] = 'z'
    pattern1.add_edge(1, 2, label='a')
    pattern1.add_edge(2, 3, label='b')

    assert utils.get_support(utils.get_embeddings(pattern1, graph)) == 1
    assert utils.get_support(utils.get_embeddings(pattern, ng)) == 2


def test_get_label_index():
    values = ('a', 'b')
    assert utils.get_label_index('a', values) == 0

    with pytest.raises(ValueError):
        utils.get_label_index('c', values)


def test_get_node_label():
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'a', 'b'

    with pytest.raises(ValueError):
        utils.get_node_label(1, 6, test)
        utils.get_node_label(2, 0, test)

    assert utils.get_node_label(1, 0, test) == 'a'


def test_get_edge_label():
    test = nx.Graph()
    test.add_nodes_from(range(1, 3))
    test.add_edge(1, 2, label='a')
    assert utils.get_edge_label(1, 2, test) == 'a'

    test.add_edge(2, 3)
    with pytest.raises(ValueError):
        utils.get_edge_label(2, 3, test)
        utils.get_edge_label(2, 4, test)


def test_is_without_edge():
    test = nx.Graph()
    test.add_node(1)
    test.add_node(2)
    assert utils.is_without_edge(test) is True

    test.add_edge(1, 2)
    assert utils.is_without_edge(test) is False


def test_get_edge_in_embedding():
    graph = init_graph_to_utils_test()['graph']
    p7 = nx.DiGraph()
    p7.add_node(1, label='x')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')

    embeddings = utils.get_embeddings(p7, graph)
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[0][0] == 1
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[0][1] == 2
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[1][0] == 1
    assert list(utils.get_edge_in_embedding(embeddings[0], p7))[1][1] == 3


def test_get_key_from_value():
    graph = init_graph_to_utils_test()['graph']
    p7 = nx.DiGraph()
    p7.add_node(1, label='x')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')

    embeddings = utils.get_embeddings(p7, graph)
    assert utils.get_key_from_value(embeddings[0], 1) == 6
    assert utils.get_key_from_value(embeddings[0], 2) == 8
    assert utils.get_key_from_value(embeddings[0], 3) == 1


def init_second_graph_to_utils_test():
    res = dict()
    g = nx.DiGraph()
    g.add_nodes_from(range(1, 9))
    g.add_edge(2, 1, label='a')
    g.add_edge(4, 1, label='a')
    g.add_edge(6, 1, label='a')
    g.add_edge(6, 8, label='a')
    g.add_edge(1, 3, label='b')
    g.add_edge(1, 5, label='b')
    g.add_edge(1, 7, label='b')
    g.nodes[1]['label'] = 'y'
    g.nodes[2]['label'] = 'x'
    g.nodes[3]['label'] = 'z'
    g.nodes[4]['label'] = 'x'
    g.nodes[5]['label'] = 'z'
    g.nodes[6]['label'] = 'x'
    g.nodes[7]['label'] = 'z'
    g.nodes[8]['label'] = 'w', 'x'
    res['g'] = g
    label_codes = LabelCodes(g)
    res['label_codes'] = label_codes
    p1 = nx.DiGraph()
    p1.add_node(1, label='x')
    p1.add_node(2)
    p1.add_edge(1, 2, label='a')
    res['p1'] = p1
    p2 = nx.DiGraph()
    p2.add_node(1, label='y')
    p2.add_node(2)
    p2.add_edge(1, 2, label='b')
    res['p2'] = p2
    ct = CodeTable(label_codes, g)
    ct.add_row(CodeTableRow(res['p1']))
    ct.add_row(CodeTableRow(res['p2']))
    ct.cover()
    res['ct'] = ct
    return res


def test_get_two_nodes_all_port():
    test = nx.DiGraph()
    test.add_node(1, is_Pattern=True)
    test.add_node(2, is_Pattern=True)
    test.add_node(3)
    test.add_node(4)
    test.add_node(5, is_Pattern=True)
    test.add_edge(1, 3)
    test.add_edge(1, 4)
    test.add_edge(2, 3)
    test.add_edge(2, 4)
    test.add_edge(5, 4)

    assert len(utils.get_two_nodes_all_port(1, 2, test)) == 2
    print('\n', utils.get_two_nodes_all_port(1, 2, test))
    assert len(utils.get_two_nodes_all_port(5, 2, test)) == 1
    print('\n', utils.get_two_nodes_all_port(5, 2, test))


def test_generate_candidates():
    res = init_second_graph_to_utils_test()
    ct = res['ct']

    candidates = utils.generate_candidates(ct.rewritten_graph(), ct)
    assert len(candidates) == 8
    assert candidates[0].first_pattern_label == 'P0'
    assert candidates[7].second_pattern_label == 'z'
    # print(candidates)


def test_compute_pattern_usage():
    res = init_second_graph_to_utils_test()
    ct = res['ct']
    """ {2: ['P0v2', 'P0v2', 'P0v2', 'P1v1', 'P1v1', 'P1v1'], 
    5: ['P0v1', 'P0v1'], 6: ['P0v2', 'wv1', 'xv1'], 
    9: ['P1v2', 'zv1'], 11: ['P1v2', 'zv1'], 
    13: ['P1v2', 'zv1']} """
    usage = utils.compute_pattern_usage(ct.rewritten_graph(), ['P0v2'], {2})
    assert usage == 3
    usage = utils.compute_pattern_usage(ct.rewritten_graph(), ['zv1'], {11, 9, 13})
    assert usage == 3

    rwg = nx.DiGraph()
    rwg.add_node(1, is_Pattern=True, label='P1')
    rwg.add_node(2)
    rwg.add_node(3, is_Pattern=True, label='P2')
    rwg.add_node(4, is_Pattern=True, label='P2')
    rwg.add_node(5)
    rwg.add_edge(1, 2, label='v1')
    rwg.add_edge(1, 5, label='v2')
    rwg.add_edge(3, 1, label='v2')
    rwg.add_edge(3, 5, label='v1')
    rwg.add_edge(4, 2, label='v1')
    usage = utils.compute_pattern_usage(rwg, ['P1v1', 'P1v2'], {2, 5})
    assert usage == 1
    # print(usage)


def test_compute_candidate_usage():
    res = init_second_graph_to_utils_test()
    ct = res['ct']
    candidates = utils.generate_candidates(ct.rewritten_graph(), ct)
    c1 = Candidate('P0', 'P0', [('v2', 'v2')])
    c1.first_pattern = res['p1']
    c1.second_pattern = res['p1']
    c1.data_port = {2}
    utils.compute_candidate_usage(ct.rewritten_graph(), c1, ct, candidates)
    assert c1.usage == 1

    c2 = Candidate('P0', 'P1', [('v2', 'v1')])
    c2.first_pattern = res['p1']
    c2.second_pattern = res['p2']
    c2.data_port = {2}
    utils.compute_candidate_usage(ct.rewritten_graph(), c2, ct, candidates)
    assert c2.usage == 3

    c3 = Candidate('z', 'P1', [('v1', 'v2')])
    c3.second_pattern = res['p2']
    with pytest.raises(ValueError):
        utils.compute_candidate_usage(ct.rewritten_graph(), c3, ct, candidates)

    c4 = Candidate('w', 'x', [('v1', 'v1')])
    c4.data_port = {6}
    utils.compute_candidate_usage(ct.rewritten_graph(), c4, ct, candidates)
    assert c4.usage == 1

    c5 = Candidate('P1', 'z', [('v2', 'v1')])
    c5.first_pattern = res['p2']
    c5.data_port = {11, 9, 13}
    utils.compute_candidate_usage(ct.rewritten_graph(), c5, ct, candidates)
    assert c5.usage == 3


def test_compute_pattern_embeddings():
    res = init_second_graph_to_utils_test()
    ct = res['ct']

    assert utils.compute_pattern_embeddings(ct.rewritten_graph(), 'P1') == 3
    assert utils.compute_pattern_embeddings(ct.rewritten_graph(), 'w') == 1


def test_is_candidate_port_exclusive():
    res = init_second_graph_to_utils_test()
    ct = res['ct']
    candidates = utils.generate_candidates(ct.rewritten_graph(), ct)
    c = Candidate('P0', 'P0', [('v1', 'v1')])
    c1 = Candidate('P0', 'P0', [('v2', 'v2')])
    utils.is_candidate_port_exclusive(candidates, c, 5)

    assert utils.is_candidate_port_exclusive(candidates, c, 5) is True
    assert utils.is_candidate_port_exclusive(candidates, c1, 2) is False


"""def test_get_candidates():
    res = init_graph2()
    ct = res['ct']
    restricted_candidates = utils.get_candidates(ct.rewritten_graph(), ct)
    assert len(restricted_candidates) == 8
    assert restricted_candidates[1].usage == 3.0
    assert restricted_candidates[6].exclusive_port_number == 3"""

""" Code Table """


def test_is_node_marked():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'x', 'w'

    with pytest.raises(ValueError):
        ct.is_node_marked(6, test, 1, 'x')
        ct.is_node_marked(1, test, 1, 'a')

    assert ct.is_node_marked(1, test, 1, 'x') is False
    test.nodes[1]['cover_mark'] = {'x': 1}
    assert ct.is_node_marked(1, test, 1, 'x') is True


def test_is_node_labels_marked():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(1)
    test.nodes[1]['label'] = 'x', 'w'

    with pytest.raises(ValueError):
        ct.is_node_labels_marked(6, test, 1, 'x')
        ct.is_node_labels_marked(1, test, 1, 'a')

    test.nodes[1]['cover_mark'] = {'x': 1}
    assert ct.is_node_labels_marked(1, test, 1, ('x', 'w')) is False
    test.nodes[1]['cover_mark'] = {'x': 1, 'w': 1}
    assert ct.is_node_labels_marked(1, test, 1, ('x', 'w')) is True


def test_is_edge_marked():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(range(1, 3))
    test.add_edge(1, 2, label='e')

    assert ct.is_edge_marked(1, 2, test, 1, 'e') is False
    test[1][2]['cover_mark'] = {'e': 1}
    assert ct.is_edge_marked(1, 2, test, 1, 'e') is True


def test_mark_node():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(1, label='x')

    with pytest.raises(ValueError):
        ct.mark_node(1, test, 1, None)

    ct.mark_node(1, test, 1, 'x')
    assert ct.is_node_marked(1, test, 1, 'x') is True

    test.nodes[1]['label'] = test.nodes[1]['label'], 'w'
    ct.mark_node(1, test, 2, ('x', 'w'))

    assert ct.is_node_marked(1, test, 1, 'x') is False
    assert ct.is_node_marked(1, test, 2, 'x') is True
    assert ct.is_node_marked(1, test, 2, 'w') is True


def test_mark_edge():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(2)
    test.add_edge(1, 2, label='e')

    with pytest.raises(ValueError):
        ct.mark_edge(1, 2, test, 1, None)
        ct.mark_edge(1, 2, test, 1, 'f')

    ct.mark_edge(1, 2, test, 1, 'e')
    assert ct.is_edge_marked(1, 2, test, 1, 'e') is True

    ct.mark_edge(1, 2, test, 2, 'e')
    assert ct.is_edge_marked(1, 2, test, 1, 'e') is False
    assert ct.is_edge_marked(1, 2, test, 2, 'e') is True


def init_second_graph_to_code_table_test():
    gtest = nx.DiGraph()
    gtest.add_nodes_from(range(1, 6))
    gtest.add_edge(1, 2, label='e')
    gtest.add_edge(2, 3, label='e')
    gtest.add_edge(2, 4, label='e')
    gtest.add_edge(5, 2, label='e')
    gtest.nodes[1]['label'] = 'A'
    gtest.nodes[2]['label'] = 'A'
    gtest.nodes[3]['label'] = 'B'
    gtest.nodes[4]['label'] = 'B'
    gtest.nodes[5]['label'] = 'A'

    pattern = nx.DiGraph()
    pattern.add_nodes_from(range(1, 3))
    pattern.add_edge(1, 2, label='e')
    pattern.nodes[1]['label'] = 'A'
    embeddings = utils.get_embeddings(pattern, gtest)
    label_codes = LabelCodes(gtest)
    code_table = CodeTable(label_codes, gtest)
    rewritten_graph = nx.DiGraph()
    return gtest, pattern, embeddings, label_codes, code_table, rewritten_graph


def test_is_embedding_marked():
    gtest = init_second_graph_to_code_table_test()[0]
    pattern = init_second_graph_to_code_table_test()[1]
    embeddings = init_second_graph_to_code_table_test()[2]
    ct = init_second_graph_to_code_table_test()[4]
    assert ct.is_embedding_marked(embeddings[0], pattern, gtest, 1) is False

    gtest[1][2]['cover_mark'] = {'e': 1}
    assert ct.is_embedding_marked(embeddings[0], pattern, gtest, 1) is True

    ct.mark_edge(1, 2, gtest, 2, 'e')
    assert ct.is_embedding_marked(embeddings[0], pattern, gtest, 1) is False
    assert ct.is_embedding_marked(embeddings[0], pattern, gtest, 2) is True


def test_mark_embedding():
    gtest = init_second_graph_to_code_table_test()[0]
    pattern = init_second_graph_to_code_table_test()[1]
    embeddings = init_second_graph_to_code_table_test()[2]
    ct = init_second_graph_to_code_table_test()[4]
    ct.mark_embedding(embeddings[0], gtest, pattern, 1)

    assert ct.is_edge_marked(1, 2, gtest, 1, 'e') is True
    assert ct.is_node_marked(1, gtest, 1, 'A') is True
    assert ct.is_node_marked(2, gtest, 1, 'A') is False

    ct.mark_embedding(embeddings[0], gtest, pattern, 2)
    assert ct.is_edge_marked(1, 2, gtest, 1, 'e') is False
    assert ct.is_node_marked(1, gtest, 1, 'A') is False

    assert ct.is_edge_marked(1, 2, gtest, 2, 'e') is True
    assert ct.is_node_marked(1, gtest, 2, 'A') is True

    pattern.nodes[2]['label'] = 'A'
    ct.mark_embedding(embeddings[0], gtest, pattern, 1)
    assert ct.is_node_marked(2, gtest, 1, 'A') is True


def test_get_node_label_number():
    test = nx.Graph()
    test.add_node(1)
    ct = init_second_graph_to_code_table_test()[4]

    with pytest.raises(ValueError):
        ct.get_node_label_number(2, test)

    test.nodes[1]['label'] = 'a'
    assert ct.get_node_label_number(1, test) == 1

    test.nodes[1]['label'] = test.nodes[1]['label'], 'b'
    assert ct.get_node_label_number(1, test) == 2

    test.add_node(2)
    assert ct.get_node_label_number(2, test) == 0


def test_search_port():
    test1 = nx.DiGraph()
    test1.add_node(40, label='x')
    test1.add_node(41)
    test1.add_node(42)
    test1.add_node(43, label='y')
    test1.add_edge(40, 41, label='a')
    test1.add_edge(40, 42, label='a')
    test1.add_edge(40, 43, label='a')
    ptest1 = nx.DiGraph()
    ptest1.add_node(1, label='x')
    ptest1.add_node(2)
    ptest1.add_edge(1, 2, label='a')
    ct = init_second_graph_to_code_table_test()[4]

    embed = utils.get_embeddings(ptest1, test1)
    ct.mark_embedding(embed[0], test1, ptest1, 1)
    ct.mark_embedding(embed[1], test1, ptest1, 1)
    port_usage = dict()
    ports = ct.search_port(test1, embed[0], 1, ptest1, port_usage)
    assert ports[0][0] == 40
    assert ports[0][1] == 1
    assert (1 in port_usage.keys()) is True
    ct.search_port(test1, embed[1], 1, ptest1, port_usage)
    assert port_usage[1] != 1


def test_is_node_edges_marked():
    gtest = init_second_graph_to_code_table_test()[0]
    pattern = init_second_graph_to_code_table_test()[1]
    embeddings = init_second_graph_to_code_table_test()[2]
    ct = init_second_graph_to_code_table_test()[4]
    ct.mark_embedding(embeddings[0], gtest, pattern, 1)

    assert ct.is_node_edges_marked(gtest, 1, pattern, 1) is True
    assert ct.is_node_edges_marked(gtest, 2, pattern, 1) is False


def test_is_node_all_labels_marked():
    ct = init_second_graph_to_code_table_test()[4]
    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')

    ct.mark_node(2, test, 1, 'x')
    ct.mark_node(1, test, 1, 'x')

    assert ct.is_node_all_labels_marked(2, test, 1) is False
    assert ct.is_node_all_labels_marked(1, test, 1) is True


def test_row_cover():
    gtest = init_second_graph_to_code_table_test()[0]
    pattern = init_second_graph_to_code_table_test()[1]
    embeddings = init_second_graph_to_code_table_test()[2]
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]

    ptest = nx.Graph()
    ptest.add_node(1, label='x')

    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')

    row_test = CodeTableRow(ptest)
    row_test.set_embeddings(utils.get_embeddings(ptest, test))
    # with pytest.raises(ValueError):
    ct.row_cover(row_test, test, 1, rewritten_graph, 0)
    assert ct.is_node_marked(1, test, 1, ptest.nodes(data=True)[1]['label']) is True
    assert ct.is_node_marked(2, test, 1, ptest.nodes(data=True)[1]['label']) is True

    row = CodeTableRow(pattern)
    row.set_embeddings(embeddings)

    ct.row_cover(row, gtest, 1, rewritten_graph, 1)

    for edge in gtest.edges(data=True):
        assert ct.is_edge_marked(edge[0], edge[1], gtest, 1, gtest[edge[0]][edge[1]]['label']) is True

    for node in gtest.nodes(data=True):
        if node[1]['label'] == 'A':
            assert ct.is_node_marked(node[0], gtest, 1, node[1]['label']) is True
        else:
            assert ct.is_node_marked(node[0], gtest, 1, node[1]['label']) is False


def test_singleton_cover():
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]

    test = nx.Graph()
    test.add_node(1, label='x')
    test.add_node(3, label='y')
    test.add_node(2)
    test.nodes()[2]['label'] = 'x', 'y'
    test.add_edge(1, 2, label='e')
    test.add_edge(2, 3, label='e')
    res = ct.singleton_cover(test, 1, rewritten_graph)

    assert res[0]['x'] == 2
    assert res[0]['y'] == 2
    assert res[1]['e'] == 2


def test_rows():
    code_table = init_second_graph_to_code_table_test()[4]
    assert len(code_table.rows()) == 0


def test_add_row():
    pattern = init_second_graph_to_code_table_test()[1]
    code_table = init_second_graph_to_code_table_test()[4]

    row1 = CodeTableRow(pattern)

    pattern2 = nx.DiGraph()
    pattern2.add_node(1, label='B')

    row2 = CodeTableRow(pattern2)

    code_table.add_row(row1)
    code_table.add_row(row2)

    assert len(code_table.rows()[0].pattern().nodes()) == 2
    assert len(code_table.rows()[1].pattern().nodes()) == 1


def test_create_rewrite_edge():
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]
    rewritten_graph.add_node(1, label='40')
    rewritten_graph.add_node(2, label='P1')

    ct.create_rewrite_edge(rewritten_graph, 2, 40, pattern_port=1)
    assert ((2, 1) in list(rewritten_graph.edges(2))) is True
    assert rewritten_graph[2][1]['label'] == 'v1'

    ct.create_rewrite_edge(rewritten_graph, 2, 41, pattern_port=2)
    assert (3 in rewritten_graph.nodes()) is True
    assert rewritten_graph.nodes[3]['label'] == '41'
    assert ((2, 3) in list(rewritten_graph.edges(2))) is True
    assert rewritten_graph[2][3]['label'] == 'v2'


def test_create_vertex_singleton_node():
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]

    rewritten_graph.add_node(1, label='40')
    ct.create_vertex_singleton_node(rewritten_graph, 'x', 40)
    assert (2 in rewritten_graph.nodes()) is True
    assert ('is_Pattern' in rewritten_graph.nodes(data=True)[2]) is True
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert ('is_singleton' in rewritten_graph.nodes(data=True)[2]) is True
    assert rewritten_graph.nodes[2]['is_singleton'] is True
    assert rewritten_graph.nodes[2]['label'] == 'x'
    assert rewritten_graph[2][1]['label'] == 'v1'


def test_create_edge_singleton_node():
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]

    rewritten_graph.add_node(1, label='40')
    ct.create_edge_singleton_node(rewritten_graph, 'a', 40, 41)
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert rewritten_graph.nodes[2]['is_singleton'] is True
    assert rewritten_graph.nodes[3]['label'] == '41'
    assert rewritten_graph.nodes[2]['label'] == 'a'
    assert rewritten_graph[2][1]['label'] == 'v1'
    assert rewritten_graph[2][3]['label'] == 'v2'


def test_create_pattern_node():
    rewritten_graph = init_second_graph_to_code_table_test()[5]
    ct = init_second_graph_to_code_table_test()[4]

    rewritten_graph.add_node(1, label='40')
    ct.create_pattern_node(rewritten_graph, 1, [(40, 1)])
    assert rewritten_graph.nodes[2]['is_Pattern'] is True
    assert ('is_singleton' in rewritten_graph.nodes(data=True)[2]) is False
    assert rewritten_graph.nodes[2]['label'] == 'P1'
    assert rewritten_graph[2][1]['label'] == 'v1'


def init_graph_to_code_table_test():
    res = dict()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, 9))
    graph.add_edge(2, 1, label='a')
    graph.add_edge(4, 1, label='a')
    graph.add_edge(6, 1, label='a')
    graph.add_edge(6, 8, label='a')
    graph.add_edge(8, 6, label='a')
    graph.add_edge(1, 3, label='b')
    graph.add_edge(1, 5, label='b')
    graph.add_edge(1, 7, label='b')
    graph.nodes[1]['label'] = 'y'
    graph.nodes[2]['label'] = 'x'
    graph.nodes[3]['label'] = 'z'
    graph.nodes[4]['label'] = 'x'
    graph.nodes[5]['label'] = 'z'
    graph.nodes[6]['label'] = 'x'
    graph.nodes[7]['label'] = 'z'
    graph.nodes[8]['label'] = 'w', 'x'

    res['graph'] = graph

    p1 = nx.DiGraph()
    p1.add_nodes_from(range(1, 4))
    p1.add_edge(1, 2, label='a')
    p1.add_edge(2, 3, label='b')
    p1.nodes[1]['label'] = 'x'
    p1.nodes[2]['label'] = 'y'
    p1.nodes[3]['label'] = 'z'
    row1 = CodeTableRow(p1)
    res['p1'] = p1
    res['row1'] = row1

    p2 = nx.DiGraph()
    p2.add_nodes_from(range(1, 3))
    p2.nodes[1]['label'] = 'x'
    p2.nodes[2]['label'] = 'x'
    p2.add_edge(1, 2, label='a')
    p2.add_edge(2, 1, label='a')
    row2 = CodeTableRow(p2)

    res['p2'] = p2
    res['row2'] = row2

    p3 = nx.DiGraph()
    p3.add_node(1, label='x')
    p3.add_node(2)
    p3.add_edge(1, 2, label='a')
    row3 = CodeTableRow(p3)
    res['p3'] = p3
    res['row3'] = row3

    p4 = nx.DiGraph()
    p4.add_node(1, label='z')
    p4.add_node(2)
    p4.add_edge(2, 1, label='b')
    row4 = CodeTableRow(p4)
    res['p4'] = p4
    res['row4'] = row4

    p5 = nx.DiGraph()
    p5.add_node(1, label='x')
    p5.add_node(2, label='y')
    p5.add_edge(1, 2, label='a')
    row5 = CodeTableRow(p5)
    res['p5'] = p5
    res['row5'] = row5

    p6 = nx.DiGraph()
    p6.add_node(1, label='y')
    p6.add_node(2, label='z')
    p6.add_edge(1, 2, label='b')
    row6 = CodeTableRow(p6)
    res['p6'] = p6
    res['row6'] = row6

    p7 = nx.DiGraph()
    p7.add_node(1, label='y')
    p7.add_node(2)
    p7.add_node(3)
    p7.add_edge(1, 2, label='a')
    p7.add_edge(1, 3, label='a')
    row7 = CodeTableRow(p7)
    res['p7'] = p7
    res['row7'] = row7
    lc = LabelCodes(graph)
    res['lc'] = lc
    ct = CodeTable(lc, graph)
    res['ct'] = ct
    return res


def test_cover():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    # ct.add_row(row7)
    ct.add_row(res['row3'])
    ct.add_row(res['row4'])
    ct.add_row(res['row5'])
    ct.add_row(res['row6'])
    ct.cover()

    # print(ct.display_ct())


"""def test_compute_ct_description_length():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    with pytest.raises(ValueError):
        ct.compute_ct_description_length()

    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 112.75

    ct.add_row(res['row1'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 113.55

    ct.add_row(res['row2'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 103.29

    ct.remove_row(res['row1'])
    ct.remove_row(res['row2'])

    ct.add_row(res['row3'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 102.23

    ct.add_row(res['row4'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 91.02

    ct.add_row(res['row5'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 108.91

    ct.add_row(res['row6'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 118.0"""


def test_rewritten_graph():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    print('\n data_port: ', ct.data_port())
    print('\n port count :', utils.count_port_node(ct.rewritten_graph()))
    # assert utils.count_port_node(ct.rewritten_graph()) == 6
    print('\n pattern infos :', utils.get_pattern_node_infos(ct.rewritten_graph()))
    # assert len(utils.get_pattern_node_infos(ct.rewritten_graph())['P0']) == 5
    print('\n port infos :', utils.get_port_node_infos(ct.rewritten_graph()))
    # can = utils.get_candidates(ct.rewritten_graph(), ct)
    # print(can)


"""def test_compute_rewritten_graph_description():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 143.51

    ct.add_row(res['row1'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 56.49

    ct.add_row(res['row2'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 38.05

    ct.remove_row(res['row1'])
    ct.remove_row(res['row2'])

    ct.add_row(res['row3'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 108.15

    ct.add_row(res['row4'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 72.84

    ct.add_row(res['row5'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 68.61

    ct.add_row(res['row6'])
    ct.cover()
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 68.61"""


def test_description_length_with_prequential_code():
    res = init_multi_graph()
    ct = res['ct']
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 291.22
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 446.34
    assert pytest.approx(ct.compute_rewritten_graph_description() + ct.description_length(), rel=1e-01) == 737.56
    ct.add_row(res['row1'])
    ct.add_row(res['row2'])
    ct.add_row(res['row3'])
    ct.cover()
    ct.compute_ct_description_length()
    assert pytest.approx(ct.description_length(), rel=1e-01) == 232.93
    assert pytest.approx(ct.compute_rewritten_graph_description(), rel=1e-01) == 108.92


def test_is_ct_edge_singleton():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    assert ct.is_ct_edge_singleton('a') is True
    assert ct.is_ct_edge_singleton('w') is False


def test_is_ct_vertex_singleton():
    res = init_graph_to_code_table_test()
    ct = res['ct']
    ct.add_row(res['row5'])
    p8 = nx.DiGraph()
    p8.add_node(1)
    p8.add_node(2, label='z')
    p8.add_edge(1, 2, label='b')
    row8 = CodeTableRow(p8)
    ct.add_row(row8)
    ct.cover()
    assert ct.is_ct_vertex_singleton('a') is False
    assert ct.is_ct_vertex_singleton('w') is True


""" Candidate"""


def test_set_usage():
    c = Candidate('P0', 'P1', [('v1', 'v2')])
    c.set_usage(2.0)

    assert c.usage == 2.0


def test_inverse():
    c = Candidate('P0', 'P1', [('v1', 'v2')])
    c_inverse = Candidate('P1', 'P0', [('v2', 'v1')])
    assert c.inverse() == c_inverse


def test_merge_candidate():
    pa = nx.DiGraph()
    pa.add_node(1, label='A')
    pa.add_node(2, label='B')
    pa.add_node(3, label='C')
    pa.add_edge(1, 2, label='a')
    pa.add_edge(2, 3, label='b')

    pb = nx.DiGraph()
    pb.add_node(1, label='D')
    pb.add_node(2, label='E')
    pb.add_node(3, label='F')
    pb.add_edge(1, 2, label='c')
    pb.add_edge(2, 3, label='d')

    c = Candidate('P0', 'P1', [('v2', 'v1')])
    c.first_pattern = pa
    c.second_pattern = pb

    graph = c.merge_candidate()

    assert len(graph.nodes()) == 5
    assert len(graph.nodes[2]['label']) == 2
    assert graph[2][4]['label'] == 'c'

    c1 = Candidate('P0', 'P0', [('v3', 'v2')])
    c1.first_pattern = pa
    c1.second_pattern = pa
    g1 = c1.merge_candidate()
    assert len(g1.nodes[3]['label']) == 2
    assert g1[4][3] is not None
    assert g1[3][5] is not None

    c2 = Candidate('P0', 'P1', [('v1', 'v1')])
    c2.first_pattern = pa
    c2.second_pattern = pb
    g2 = c2.merge_candidate()
    assert len(g2.nodes[1]['label']) == 2
    assert g2[1][4]['label'] == 'c'

    c3 = Candidate('P0', 'P1', [('v1', 'v1'), ('v3', 'v3')])
    c3.first_pattern = pa
    c3.second_pattern = pb
    g3 = c3.merge_candidate()
    assert len(g3.nodes()) == 4
    assert g3.nodes[1]['label'] == ('D', 'A')
    assert g3.nodes[3]['label'] == ('F', 'C')
    assert g3[4][3] is not None

    p1 = nx.DiGraph()
    p1.add_node(1, label='C')
    p1.add_node(2)
    p1.add_node(3, label='C')
    p1.add_edge(1, 2, label='single')
    p1.add_edge(3, 1, label='single')

    p2 = nx.DiGraph()
    p2.add_node(1, label='x')
    p2.add_node(2)
    p2.add_edge(1, 2, label='a')
    c4 = Candidate('P1', 'P1', [('v2', 'v3'), ('v3', 'v2'), ('v1', 'v1')])
    c4.first_pattern = p1
    c4.second_pattern = p1
    g4 = c4.merge_candidate()
    assert len(g4.nodes()) == 3


def test_final_pattern():
    pa = nx.DiGraph()
    pa.add_node(1, label='A')
    pa.add_node(2, label='B')
    pa.add_node(3, label='C')
    pa.add_edge(1, 2, label='a')
    pa.add_edge(2, 3, label='b')

    pb = nx.DiGraph()
    pb.add_node(1, label='D')
    pb.add_node(2, label='E')
    pb.add_node(3, label='F')
    pb.add_edge(1, 2, label='c')
    pb.add_edge(2, 3, label='d')
    c2 = Candidate('P0', 'P1', [('v1', 'v1')])
    c2.first_pattern = pa
    c2.second_pattern = pb
    g2 = c2.final_pattern()
    assert len(g2.nodes[1]['label']) == 2
    assert g2[1][4]['label'] == 'c'


""" GraphMDL test"""
g = nx.DiGraph()
g.add_nodes_from(range(1, 9))
g.add_edge(2, 1, label='a')
g.add_edge(4, 1, label='a')
g.add_edge(6, 1, label='a')
g.add_edge(6, 8, label='a')
g.add_edge(8, 6, label='a')
g.add_edge(1, 3, label='b')
g.add_edge(1, 5, label='b')
g.add_edge(1, 7, label='b')
g.nodes[1]['label'] = 'y'
g.nodes[2]['label'] = 'x'
g.nodes[3]['label'] = 'z'
g.nodes[4]['label'] = 'x'
g.nodes[5]['label'] = 'z'
g.nodes[6]['label'] = 'x'
g.nodes[7]['label'] = 'z'
g.nodes[8]['label'] = 'w', 'x'


def test_fit():
    with pytest.raises(ValueError):
        GraphMDL().fit(None)

    mdl = GraphMDL()
    mdl.fit(g)
    # mdl.summary()
    assert mdl.description_length() != 0.0

    assert GraphMDL().fit(g, timeout=0.01).description_length() != 0


def test_patterns():
    assert len(GraphMDL().fit(g).patterns()) == 3


def test_description_length():
    assert pytest.approx(GraphMDL().fit(g).description_length(), rel=1e-01) == 144.8


def test_initial_description_length():
    assert pytest.approx(GraphMDL().fit(g).initial_description_length(), rel=1e-01) == 256.3


def test_graphmdl_on_multidigraph():
    graph = init_multi_graph()['graph']
    GraphMDL().fit(graph)
