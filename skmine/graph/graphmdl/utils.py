"""  Function Utilities  """
import logging
import math
from collections import Counter

import networkx as nx
from networkx import Graph
from networkx.algorithms import isomorphism as iso

from skmine.graph.graphmdl.candidate import Candidate


def log2(value, total):
    """ Compute logarithm in the binary base
    Parameters
    ----------
    value
    total

    Returns
    -------
    float
    """
    return -math.log2(value / total)


def prequential_code(seq, epsilon=0.5):
    """ Compute the prequential code for a given sequence according an special epsilon
    Parameters
    -----------
    seq
    epsilon
    Returns
    -------
    float
    """
    code = 0.0
    eps = epsilon
    total_epsilon = len(seq) * epsilon
    for value in seq:
        for i in range(0, value):
            code += log2(eps, total_epsilon)
            total_epsilon += 1
            eps += 1

        eps = epsilon
    return code


def count_edge_label(graph: Graph):
    """ Count each edge label occurrence in a graph and store the result
    Parameters
    ----------
    graph:Graph
        the data graph
    Returns
    -------
    dict
    """

    edges = dict()
    j = 1
    for u, v, d in graph.edges(data=True):
        if 'label' in d:
            if type(d['label']) is tuple:
                for i in d['label']:
                    edges[j] = i
                    j = j + 1
            else:
                edges[j] = d['label']
                j += 1
    return dict([(i, j) for i, j in Counter(edges.values()).items()])


def count_vertex_label(graph: Graph):
    """ Count each vertex label occurrence in a graph and store the result
        Parameters
        ----------
        graph:Graph
            the data graph
        Returns
        -------
        dict
      """
    vertex = dict()
    j = 1
    for u, d in graph.nodes(data=True):
        if 'label' in d:
            if type(d['label']) is tuple:
                for i in d['label']:
                    vertex[j] = i
                    j = j + 1
            else:
                vertex[j] = d['label']
                j += 1
    return dict([(i, j) for i, j in Counter(vertex.values()).items()])


def get_total_label(graph: Graph):
    """ Compute total number of labels in the graph
       Parameters
       ----------
       graph:Graph
         the data graph
       Returns
       -------
         float
    """

    labels = count_edge_label(graph)
    labels.update(count_vertex_label(graph))
    total = 0.0  # The total number of labels in the graph
    for i in labels.values():
        total = total + i

    return total


def binomial(n, k):
    """ Compute the binomial coefficient for a given n and given k. Also called "n choose k"
    Parameters
    ----------
    n
    k
    Returns
    -------
    float
    """
    if k > n:
        raise ValueError(f"{k} should be lower than {n} in binomial coefficient")
    elif k == 0:
        return 1
    elif k > n / 2:
        return binomial(n, n - k)

    return n * binomial(n - 1, k - 1) / k


def universal_integer_encoding(x):
    """ Compute universal codeword sets and representation for integer except 1
    Parameters
    ----------
    x
    Returns
    -------
    int
    """
    if x < 1:
        raise ValueError(f"{x} should be higher than 1")
    else:
        return math.floor(math.log2(x)) + 2 * math.floor(math.log2(math.floor(math.log2(x)) + 1)) + 1


def universal_integer_encoding_with0(x):
    """ Compute universal codeword sets and representation for integer
    Parameters
    ----------
    x
    Returns
    -------
    int
    """
    if x < 0:
        raise ValueError(f"{x} should be higher than 0")
    else:
        return universal_integer_encoding(x + 1)


def encode(pattern: Graph, standard_table):
    """ Compute a given graph description length according a given label codes
    Parameters
    ----------
    standard_table : The label codes
    pattern : Graph
        The given graph
    Returns
    -------
    float
    """
    edges = count_edge_label(pattern)  # count each pattern edge label occurrences
    vertex = count_vertex_label(pattern)  # count each pattern vertex label occurrences

    # Get total number of label in the standard table
    total_label = len(standard_table.vertex_lc()) + len(standard_table.edges_lc())
    vertex_number = len(pattern.nodes())

    total_label_description = math.log2(total_label)  # description length for all labels
    vertex_number_description = universal_integer_encoding_with0(vertex_number)  # description length for all vertex

    # Compute description length for vertex
    vertex_description = dict()
    for u, v in vertex.items():
        desc = standard_table.vertex_lc()[u] + universal_integer_encoding_with0(v) + math.log2(
            binomial(vertex_number, v))
        vertex_description[u] = desc

    # Compute description length for edges
    edges_description = dict()
    for a, b in edges.items():
        desc = standard_table.edges_lc()[a] + universal_integer_encoding_with0(b) + math.log2(
            binomial(int(math.pow(vertex_number, 2)), b))
        edges_description[a] = desc

    # Compute description length through description length of edges and vertex
    description_length = 0.0
    for i in vertex_description.values():
        description_length = description_length + i
    for j in edges_description.values():
        description_length = description_length + j

    return description_length + total_label_description + vertex_number_description


def encode_vertex_singleton(standard_table, vertex_label):
    """ Compute a vertex singleton description length
        Parameters
        ----------
        standard_table : a labels codes
        vertex_label : The vertex singleton label
        Returns
        -------
        float
    """
    if vertex_label == "" or vertex_label is None:
        raise ValueError(f"You should give a vertex label")
    else:
        # Get total number of label in the standard table
        total_label = len(standard_table.vertex_lc()) + len(standard_table.edges_lc())
        total_label_description = math.log2(total_label)  # description length for all labels
        vertex_number_description = universal_integer_encoding_with0(1)  # description length for all vertex

        # Compute description length for vertex
        desc = standard_table.vertex_lc()[vertex_label] + universal_integer_encoding_with0(1) + math.log2(
            binomial(1, 1))

        return desc + total_label_description + vertex_number_description


def encode_edge_singleton(standard_table, edge_label):
    """ Compute an edge singleton description length
        Parameters
        ----------
        standard_table : a labels codes
        edge_label : the edge singleton label
        Returns
        -------
        float
    """
    if edge_label == "" or edge_label is None:
        raise ValueError("You should give an edge label")
    else:
        # Get total number of label in the standard table
        total_label = len(standard_table.vertex_lc()) + len(standard_table.edges_lc())
        total_label_description = math.log2(total_label)  # description length for all labels
        vertex_number_description = universal_integer_encoding_with0(2)  # description length for all vertex

        # Compute description length for vertex
        desc = standard_table.edges_lc()[edge_label] + universal_integer_encoding_with0(1) + math.log2(binomial(4, 1))

        return desc + total_label_description + vertex_number_description


def encode_singleton(standard_table, arity, label):
    """ Compute a singleton description length according her arity
        Parameters
        ----------
        standard_table : a label codes
        label : The singleton label
        arity : 1 for the vertex singleton and 2 for the edge singleton
        Returns
        -------
        float
    """

    if arity == 1:
        return encode_vertex_singleton(standard_table, label)
    elif arity == 2:
        return encode_edge_singleton(standard_table, label)
    else:
        raise ValueError("arity should must be 1 or 2")


def _node_match(node1, node2):
    """ Compare two given nodes
    Parameters
    ---------
    node1
    node2
    Returns
    -------
    bool
    """
    if 'label' in node1 and 'label' in node2:
        res = list()
        if type(node1['label']) is str and type(node2['label']) is str:
            res.append(node1['label'] == node2['label'])
        elif type(node1['label']) is not str and type(node2['label']) is str:
            res.append(node2['label'] in node1['label'])
        elif type(node1['label']) is not str and type(node2['label']) is not str:
            for i in node2['label']:
                res.append(i in node1['label'])
        else:
            res.append(False)

        return not (False in res)
    elif 'label' not in node1 and 'label' in node2:
        return False
    else:
        return True


def _edge_match(edge1, edge2):
    """ Compare two given edges
    Parameters
    ----------
    edge1
    edge2
    Returns
    -------
    bool
    """
    if 'label' in edge1 and 'label' in edge2:
        return edge1['label'] == edge2['label']
    else:
        return ValueError("All edges must have labels")


def _edge_match_for_multigraph(edge1, edge2):
    """ Compare two given nodes in a given multigraph
    Parameters
    ----------
    edge1
    edge2
    Returns
    -------
    bool
    """
    edge1_values = dict(edge1).values()
    edge2_values = dict(edge2).values()
    edge1_labels = [list(edge1_values)[0][i] for i in list(edge1_values)[0].keys() if i == 'label']
    edge2_labels = [list(edge2_values)[0][i] for i in list(edge2_values)[0].keys() if i == 'label']
    if len(edge2_labels) != 0 and len(edge1_labels) != 0:
        return not (False in [label in edge1_labels for label in edge2_labels])
    else:
        raise ValueError("All edges must have labels")


def get_embeddings(pattern, graph):
    """ Provide the embeddings of a pattern in a given graph
    Parameters
    ----------
    pattern
    graph
    Returns
    -------
    list
    """

    # Create functions to compare node and edge label
    comp = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }

    comp_multigraph = {
        'node_match': _node_match,
        'edge_match': _edge_match_for_multigraph
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)
    if type(graph) is nx.DiGraph:
        graph_matcher = iso.DiGraphMatcher(graph, pattern, **comp)
    elif type(graph) is nx.MultiDiGraph:
        graph_matcher = iso.MultiDiGraphMatcher(graph, pattern, **comp_multigraph)
    elif type(graph) is nx.MultiGraph:
        graph_matcher = iso.MultiGraphMatcher(graph, pattern, **comp_multigraph)
    else:
        graph_matcher = iso.GraphMatcher(graph, pattern, **comp)
    return list(graph_matcher.subgraph_monomorphisms_iter())


def is_vertex_singleton(pattern):
    """ Check if a given pattern is a vertex singleton pattern
    Parameters
    ---------
    pattern
    Returns
    ---------
    bool
    """
    return len(pattern.nodes()) == 1 and get_total_label(pattern) == 1


def is_edge_singleton(pattern):
    """ Check if a given pattern is an edge singleton pattern
    Parameters
    ---------
    pattern
    Returns
    ---------
    bool
    """
    if len(pattern.nodes()) == 2:
        # Check first if the nodes haven't labels
        if "label" not in pattern.nodes(data=True)[1] \
                and "label" not in pattern.nodes(data=True)[2]:
            # Check if the edge have exactly one label
            if count_edge_label(pattern) is not None \
                    and len(count_edge_label(pattern).values()) == 1 \
                    and list(count_edge_label(pattern).values())[0] == 1:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def get_support(embeddings):
    """ Compute the pattern support in the graph according a minimum image based support

    The minimum image-based support is the minimum number of graph nodes,
    that a pattern node can replace

    Parameters
    ---------
    embeddings
    Returns
    -------
    int
    """
    if len(embeddings) != 0:
        node_embeddings = dict()
        # Compute for each pattern node, the graph nodes who can replace,
        # and store it in a dictionary
        for e in embeddings:
            for i in e.items():
                if i[1] in node_embeddings:
                    node_embeddings[i[1]].add(i[0])
                else:
                    node_embeddings[i[1]] = set()
        embed = dict()
        # Compute for each pattern node,the total number of graph nodes who could replace,
        # and store it in a dictionary
        for key, value in node_embeddings.items():
            embed[key] = len(value)

        return min(embed.values())
    else:
        return 0


def get_label_index(label, values):
    """ Provide a label index in the values
    Parameters
    ----------
    label
    values
    Returns
    ----------
    int
    """
    if label in values:
        return values.index(label)
    else:
        raise ValueError(f"{label} should be in the {values}")


def get_node_label(key, index, graph):
    """ Provide a particular node label in a given graph by the index and the node key
    Parameters
    ----------
    key
    index
    graph
    Returns
    ---------
    str
    """
    if key in graph.nodes():
        if len(graph.nodes(data=True)[key]['label']) > index and type(graph.nodes(data=True)[key]['label']) is tuple:
            return graph.nodes(data=True)[key]['label'][index]
        else:
            return graph.nodes(data=True)[key]['label']
    else:
        raise ValueError(f"{index} shouldn't be out of bounds and {key} should be a graph node")


def get_edge_label(start, end, graph):
    """ Provide a particular edge label in a given graph by the edge start and the edge end
    Parameters
    ---------
    start
    end
    graph
    Returns
    ---------
    str
    """
    if start in graph.nodes() and end in graph.nodes():
        if type(graph) is nx.MultiDiGraph:
            if (start, end) in list(graph.edges(start)):
                if not (False in ['label' in v
                                  for v in graph.get_edge_data(start, end).values()]):
                    return [v['label'] for v in graph.get_edge_data(start, end).values()]
            elif (end, start) in list(graph.edges(end)):
                if not (False in ['label' in v
                                  for v in graph.get_edge_data(end, start).values()]):
                    return [v['label'] for v in graph.get_edge_data(end, start).values()]
            else:
                raise ValueError(f"{start}-{end} should be a graph edge and should have a label")
        else:
            if (start, end) in list(graph.edges(start)):
                if 'label' in graph[start][end]:
                    return graph[start][end]['label']
            elif (end, start) in list(graph.edges(end)):
                if 'label' in graph[end][start]:
                    return graph[end][start]['label']
            else:
                raise ValueError(f"{start}-{end} should be a graph edge and should have a label")
    else:
        raise ValueError(f"{start} and {end} should be a graph nodes")


def is_without_edge(pattern):
    """ Check if the pattern is without edge
    Parameters
    ----------
    pattern
    Returns
    ---------
    bool
    """
    return len(pattern.edges()) == 0


def _get_node_labels(node, graph):
    """ Provide the labels for a particular graph node,
    if the node doesn't have labeled, return the node

    Parameters
    ----------
    node
    graph
    Returns
    --------
    str
    """
    if 'label' in graph.nodes[node]:
        return graph.nodes[node]['label']
    else:
        return node


def display_graph(graph: Graph):
    """ Display a given graph in a specific string sequence
    Parameters
    ---------
    graph
    Returns
    --------
    str
    """
    msg = ""
    if len(graph.edges()) != 0:
        for edge in graph.edges(data=True):
            if "label" in edge[2]:
                msg += "{}--{}-->{}".format(_get_node_labels(edge[0], graph),
                                            edge[2]['label'], _get_node_labels(edge[1], graph)) + "\n"
            else:
                msg += "{}--->{}".format(_get_node_labels(edge[0], graph),
                                         _get_node_labels(edge[1], graph)) + "\n"
    else:
        msg += "{}".format(_get_node_labels(1, graph))
    return msg


def draw_graph(g: Graph):
    """ Draw a given pattern
    Parameters
    ----------
    g
    """
    pos = nx.spring_layout(g, seed=7)
    edge_labels = dict([((u, v,), d['label']) for u, v, d in g.edges(data=True)])
    node_labels = dict([(u, d['label']) for u, d in g.nodes(data=True) if 'label' in d])
    nx.draw_networkx(g, pos, with_labels=True)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red", font_weight="bold", font_size=14)
    nx.draw_networkx_labels(g, pos, node_labels, font_color="red", font_weight="bold", font_size=14)


def get_edge_in_embedding(embedding, pattern):
    """ Provide the pattern edges who are in a given embedding
    Parameters
    ----------
    embedding
    pattern
    Returns
    -------
    set
    """
    keys = list(embedding.keys())
    values = list(embedding.values())
    edges = set()
    i = 0
    while i <= len(keys) - 1:
        j = i
        pattern_edges = list(pattern.edges())
        while j <= len(keys) - 1:
            if (values[i], values[j]) in pattern_edges:
                edges.add((values[i], values[j]))
            if (values[j], values[i]) in pattern_edges:
                edges.add((values[j], values[i]))
            j += 1
        del pattern_edges
        i += 1
    return edges


def get_key_from_value(data, value):
    """ Provide a dictionary key from the value
    Parameters
    ----------
    data
    value
    Returns
    --------
    int
    """
    return [k for k, v in data.items() if v == value][0]


def get_two_nodes_all_port(node1, node2, rewritten_graph):
    """ Provide all port in a graph for a potential candidate nodes
     Parameters
     ----------
     node1
     node2
     rewritten_graph
     Returns
     -------
     list
     """
    # check for all port edges if the node 1 and the node2 are port neighbors
    res = set()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            edge1 = (node1, node[0])
            edge2 = (node2, node[0])
            cpt = 0
            for edge in rewritten_graph.in_edges(node[0]):
                if edge == edge1 or edge == edge2:
                    cpt += 1
            if cpt == 2:
                res.add(node[0])
    return list(res)


def get_all_candidate_ports_labels_tuple(rewritten_graph, ports, first_node, second_node, inverse=False):
    """ Provide a port tuples for a given two nodes who are a candidate
    Parameters
    ---------
    rewritten_graph
    ports : a complete list of data port
    first_node
    second_node
    inverse : boolean to decide for the tuple elements positions

    Returns
    -------
    list"""
    port = []
    for p in ports:
        first_port = get_edge_label(first_node, p, rewritten_graph)
        second_port = get_edge_label(second_node, p, rewritten_graph)
        if not inverse:
            port.append((first_port, second_port))
        else:
            port.append((second_port, first_port))
    return port


def is_isomorphic(graph, pattern):
    """ Check if two graph is isomorphic
    Parameters
    ---------
    graph
    pattern
    Returns
    -------
    bool
    """

    opt = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, pattern, **opt)
    else:
        graph_matcher = iso.GraphMatcher(graph, pattern, **opt)

    return graph_matcher.is_isomorphic()


def get_automorphisms(graph):
    """ Provide a given graph automorphisms
    Parameters
    ----------
    graph
    Returns
    ---------
    list
    """
    opt = {
        'node_match': _node_match,
        'edge_match': _edge_match
    }
    graph_matcher = None
    # Create matcher according the graph type (directed or no)

    if nx.is_directed(graph):
        graph_matcher = iso.DiGraphMatcher(graph, graph, **opt)
    else:
        graph_matcher = iso.GraphMatcher(graph, graph, **opt)

    """automorphisms = set()
    for auto in list(graph_matcher.isomorphisms_iter())[1:]:
        for i, j in auto.items():
            if i != j:
                automorphisms.add((i, j))"""

    return list(graph_matcher.isomorphisms_iter())


def get_port_candidates(patterns_list):
    """ Provide a list of node tuple from the given list
    Parameters
    ----------
    patterns_list
    Returns
    -------
    list
    """
    res = set()
    i = 0
    while i <= len(patterns_list) - 1:
        j = i + 1
        while j <= len(patterns_list) - 1:
            res.add((patterns_list[i], patterns_list[j]))
            j += 1
        i += 1
    return list(res)


def compute_pattern_usage(rewritten_graph, pattern_format, ports):
    """ Compute pattern usage in the rewritten graph
    Parameters
    ---------
    rewritten_graph
    pattern_format: The pattern label concatenated with the port label
    ports : The pattern candidates ports
    Returns
    -------
    float
    """
    pattern_usage = 0
    ports_infos = get_port_node_infos(rewritten_graph)
    # compute for each given ports, the information about his pattern neighbors
    for port in ports:
        infos = ports_infos[port]
        # compute how many times, the pattern is the port neighbors
        for i in infos:
            if i in pattern_format:
                pattern_usage += 1
        # If the pattern have different appearances as neighbors,
        # return the pattern total usage sliced by the number of appearances
    return pattern_usage / len(pattern_format)


def compute_candidate_usage(rewritten_graph, candidate, code_table, candidates):
    """ Compute an estimated usage for a  given candidate
    Parameters
    -----------
    rewritten_graph
    candidate
    code_table : a Code Table
    candidates :candidates list
    """
    p1_format = []  # format of the first candidate pattern with its ports
    p2_format = []  # format of the second candidate pattern with its ports
    for p in candidate.port:
        p1_format.append(candidate.first_pattern_label + p[0])
        p2_format.append(candidate.second_pattern_label + p[1])

    # Compute each candidate patterns usage
    first_pattern_usage = compute_pattern_usage(rewritten_graph, p1_format, list(candidate.data_port))
    second_pattern_usage = compute_pattern_usage(rewritten_graph, p2_format, list(candidate.data_port))

    # Compute the candidate estimated usage according the candidate particularities
    if candidate.first_pattern_label == candidate.second_pattern_label and \
            (candidate == candidate.inverse() or candidate.inverse() not in candidates):
        candidate.set_usage(int(first_pattern_usage / 2))
    elif candidate.first_pattern is not None and candidate.second_pattern is not None:
        if is_without_edge(candidate.first_pattern) and not is_without_edge(candidate.second_pattern):
            # if only the second pattern doesn't have edges
            candidate.set_usage(second_pattern_usage)
        elif is_without_edge(candidate.second_pattern) and not is_without_edge(candidate.first_pattern):
            # if only the second pattern doesn't have edges
            candidate.set_usage(first_pattern_usage)
        elif is_without_edge(candidate.first_pattern) and is_without_edge(candidate.second_pattern):
            candidate.set_usage(compute_pattern_embeddings(rewritten_graph, candidate.first_pattern_label))
        else:
            # if both of the pattern have edges
            candidate.set_usage(min(first_pattern_usage, second_pattern_usage))

    elif candidate.second_pattern is not None and candidate.first_pattern is None:
        raise ValueError("The singleton should be the second pattern")
    elif candidate.first_pattern is not None and candidate.second_pattern is None:

        # A pattern with a singleton
        if code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # the singleton is an edge singleton
            if is_without_edge(candidate.first_pattern):
                # if the pattern doesn't have edge
                candidate.set_usage(second_pattern_usage)
            else:
                candidate.set_usage(min(first_pattern_usage, second_pattern_usage))
        else:
            # The singleton is a vertex singleton
            candidate.set_usage(first_pattern_usage)

    else:
        # Only singleton case
        if code_table.is_ct_edge_singleton(candidate.first_pattern_label) \
                and not code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # only first pattern is an edge singleton
            candidate.set_usage(first_pattern_usage)
        elif not code_table.is_ct_edge_singleton(candidate.first_pattern_label) \
                and code_table.is_ct_edge_singleton(candidate.second_pattern_label):
            # only second pattern is an edge singleton
            candidate.set_usage(second_pattern_usage)
        else:
            # both singleton is vertex singleton
            candidate.set_usage(min(first_pattern_usage, second_pattern_usage))


def compute_pattern_embeddings(rewritten_graph, pattern):
    """ Compute pattern embeddings in the rewritten graph
    Parameters
    ----------
    rewritten_graph
    pattern
    Returns
    --------
    int
    """
    res = 0
    for node in rewritten_graph.nodes(data=True):
        if node[1]['label'] == pattern:
            res += 1
    return res


def is_candidate_port_exclusive(candidates, candidate, port):
    """ Check if a given  port are neighbors who are not the candidate nodes number
     Parameters
     -----------
     candidates : Candidates list
     candidate
     port : The given port
     Returns
     ----------
     bool
     """
    res = []
    # search in the candidates list if there is one
    # who are the given candidate data port as his data port
    for c in candidates:
        for p in c.data_port:
            if p == port:
                res.append(c == candidate)

    return not (False in res)


def generate_candidates(rewritten_graph, code_table):
    """ Search in the rewritten graph, the pattern who share a same port
    Parameters
    ----------
    rewritten_graph
    code_table
    Returns
    ----------
    list
    """
    candidates = list()
    ports = [node[0] for node in rewritten_graph.nodes(data=True) if 'is_Pattern' not in node[1]]
    for port in ports:
        patterns = [edge[0] for edge in rewritten_graph.in_edges(port)]  # the port neighbor
        port_candidates = get_port_candidates(patterns)
        # del patterns
        for c in port_candidates:
            first_pattern = rewritten_graph.nodes[c[0]]
            second_pattern = rewritten_graph.nodes[c[1]]
            all_candidate_port = get_two_nodes_all_port(c[0], c[1], rewritten_graph)

            # respect the candidate pattern order

            # If both candidate element are the pattern
            if 'is_singleton' not in first_pattern and 'is_singleton' not in second_pattern:

                if int(first_pattern['label'].split('P')[1]) < int(second_pattern['label'].split('P')[1]):
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0], c[1])
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                else:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0], c[1], True)
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)

                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
                candidate.second_pattern = code_table.rows()[
                    int(candidate.second_pattern_label.split('P')[1])].pattern()

            # if one node is a pattern and the second a singleton
            elif 'is_singleton' in first_pattern and 'is_singleton' not in second_pattern:
                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0], c[1], True)
                candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
                # candidate.second_pattern = create_singleton_pattern(candidate.second_pattern_label, code_table)

            elif 'is_singleton' not in first_pattern and 'is_singleton' in second_pattern:
                ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0],
                                                             c[1])
                candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                candidate.first_pattern = code_table.rows()[int(candidate.first_pattern_label.split('P')[1])].pattern()
                candidate.second_pattern = create_singleton_pattern(candidate.second_pattern_label, code_table)

            # if both of the candidate pattern are singletons
            else:  # if both of the candidate elements are the singleton
                if first_pattern['label'] < second_pattern['label']:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0], c[1])
                    candidate = Candidate(first_pattern['label'], second_pattern['label'], ports)
                else:
                    ports = get_all_candidate_ports_labels_tuple(rewritten_graph, all_candidate_port, c[0], c[1], True)
                    candidate = Candidate(second_pattern['label'], first_pattern['label'], ports)

            # Store all candidates' data_port to see if there are exclusive port
            for p in all_candidate_port:
                candidate.data_port.add(p)

            if candidate not in candidates:
                candidates.append(candidate)
            else:
                c = candidates[candidates.index(candidate)]
                for p in candidate.data_port:
                    c.data_port.add(p)

    for r in candidates:
        compute_candidate_usage(rewritten_graph, r, code_table, candidates)
        exclusive_port_number = 0
        # Search exclusive port
        for port in r.data_port:
            if is_candidate_port_exclusive(candidates, r, port):
                exclusive_port_number += 1
        r.exclusive_port_number = exclusive_port_number

        # create singleton pattern
        if r.first_pattern is None and r.second_pattern is not None:
            raise ValueError("The second pattern should be the singleton")
        elif r.first_pattern is not None and r.second_pattern is None:
            r.second_pattern = create_singleton_pattern(r.second_pattern_label, code_table)
        elif r.first_pattern is None and r.second_pattern is None:
            r.first_pattern = create_singleton_pattern(r.first_pattern_label, code_table)
            r.second_pattern = create_singleton_pattern(r.second_pattern_label, code_table)
    return candidates


def create_singleton_pattern(label, code_table):
    """ Create a graph who represent a singleton
    Parameters
    ----------
    label
    code_table
    Returns
    -------
    Graph
    """
    if code_table.data_is_multigraph():
        pattern = nx.MultiDiGraph()
    else:
        pattern = nx.DiGraph()

    if code_table.is_ct_edge_singleton(label):
        pattern.add_nodes_from(range(1, 3))
        pattern.add_edge(1, 2, label=label)
    elif code_table.is_ct_vertex_singleton(label):
        pattern.add_node(1, label=label)
    else:
        raise ValueError("The label should be a vertex or an edge label")
    return pattern


def count_port_node(rewritten_graph):
    """ Count the port number in a rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    --------
    int
    """
    numb = 0
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            numb += 1
    return numb


def get_pattern_node_infos(rewritten_graph):
    """ Provide pattern node information from the rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    --------
    dict
    """
    pattern_node = dict()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' in node[1]:
            res = []
            for edge in rewritten_graph.edges(node[0], data=True):
                res.append(edge[2]['label'])

        if node[1]['label'] in pattern_node:
            pattern_node[node[1]['label']].append(res)
        else:
            pattern_node[node[1]['label']] = []
            pattern_node[node[1]['label']].append(res)

        if 'is_singleton' in node[1]:
            if node[1]['is_singleton'] is True:
                if 'singleton' not in pattern_node[node[1]['label']]:
                    pattern_node[node[1]['label']].append('singleton')
            else:
                raise ValueError("is_singleton should be true or shouldn't exist")

    return pattern_node


def get_port_node_infos(rewritten_graph):
    """ Provide port node information from the rewritten graph
    Parameters
    ----------
    rewritten_graph
    Returns
    -------
    dict
    """
    port_node = dict()
    for node in rewritten_graph.nodes(data=True):
        if 'is_Pattern' not in node[1]:
            for edge in rewritten_graph.in_edges(node[0], data=True):
                if node[0] in port_node:
                    port_node[node[0]].append(rewritten_graph.nodes(data=True)[edge[0]]['label'] + edge[2]['label'])
                else:
                    port_node[node[0]] = []
                    port_node[node[0]].append(rewritten_graph.nodes(data=True)[edge[0]]['label'] + edge[2]['label'])
    return port_node


def get_port_node(rewritten_graph, node):
    """ Provide an embedding vertex port
    Parameters
    ----------
    rewritten_graph
    node
    Returns
    --------
    set
    """
    res = set()
    for edge in rewritten_graph.out_edges(node, data=True):
        res.add(int(edge[2]['label'].split('v')[1]))
    return res


def get_graph_from_file(file):
    """ Construct a networkx graph from a given file
    Parameters
    ----------
    file
    Returns
    -------
    Graph
    """
    file = open(file)
    lines = [line.split(' ') for line in file][3:]
    graph = nx.DiGraph()
    for line in lines:
        if line[0] == 'v':
            graph.add_node(int(line[1]) + 1, label=line[3].split('\n')[0])
        else:
            graph.add_edge(int(line[1]) + 1, int(line[2]) + 1, label=line[3].split('\n')[0])
    file.close()
    return graph


""" GraphMDL logger"""
MyLogger = logging.getLogger("skmine.graph.graphmdl")
