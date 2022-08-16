import networkx as nx


class Candidate:
    """
        It represents candidate whose format is <P1,P2,{(v1,v1)..}> in the algorithm,
        here, P1 and P2 are patterns and the last parameters, a port list

        Parameters
        -----------
        first_pattern_label : str
            It's the candidate first pattern name
        second_pattern_label : str
            It's the candidate second pattern name
        port: list
            It's the candidate port list
    """

    def __init__(self, first_pattern_label, second_pattern_label, port):
        self.first_pattern_label = first_pattern_label
        self.second_pattern_label = second_pattern_label
        self.first_pattern = None
        self.second_pattern = None
        self.port = port  # list of the candidate port association
        self.data_port = set()  # candidate port number in the rewritten graph
        self.usage = 0  # estimated usage
        self.exclusive_port_number = 0
        self._final_pattern = None  # merge pattern
        self.code_length = 0.0  # merge pattern description length

    def set_usage(self, usage):
        """ Set candidate estimated usage
        Parameters
        ----------
        usage
        """
        self.usage = usage

    def final_pattern(self):
        """ Provide the candidate merge pattern
        Returns
        --------
        object
        """
        if self._final_pattern is None:
            self._final_pattern = self.merge_candidate()
            return self._final_pattern
        else:
            return self._final_pattern

    def merge_candidate(self):
        """ Merge a candidate pattern
        Returns
        --------
        Graph
        """
        if type(self.first_pattern) is nx.MultiDiGraph:
            graph = nx.MultiDiGraph()
        else:
            graph = nx.DiGraph()
        ports = []
        for p in self.port:
            port = (int(p[0].split('v')[1]), int(p[1].split('v')[1]))
            ports.append(port)
        # Create first pattern node and edges
        self.create_candidate_first_pattern(self.first_pattern, graph)
        # Create second pattern node and edges
        self.create_candidate_second_pattern(self.second_pattern, graph, ports)
        return graph

    def create_candidate_first_pattern(self, pattern, graph):
        """ Create a given candidate first pattern nodes and edges
        Parameters
        ---------
        pattern
        graph
        """
        # create the pattern nodes with their labels in the given graph
        for node in pattern.nodes(data=True):
            if 'label' in node[1]:
                graph.add_node(node[0], label=node[1]['label'])
            else:
                graph.add_node(node[0])
        # create the pattern edges with their labels  in the given graph
        for edge in pattern.edges(data=True):
            graph.add_edge(edge[0], edge[1], label=edge[2]['label'])

    def create_candidate_second_pattern(self, pattern, graph, ports):
        """ Create a given candidate second pattern nodes and edges,
            and connected it with the existent graph elements by the port
            Parameters
            ---------
            pattern
            graph
            ports
        """
        mapping = dict()  # Mapping between pattern nodes and graph nodes number
        second_port = [p[1] for p in ports]  # pattern ports number
        first_port = [p[0] for p in ports]  # graph ports number

        # create the non-port pattern nodes with their labels in the graph
        # for the pattern port node add only the labels if it's necessary
        for node in pattern.nodes(data=True):
            if node[0] not in second_port:
                mapping[node[0]] = len(graph.nodes()) + 1
                if 'label' in node[1]:
                    graph.add_node(len(graph.nodes()) + 1, label=node[1]['label'])
                else:
                    graph.add_node(len(graph.nodes()) + 1)
            else:
                index = second_port.index(node[0])
                if 'label' in node[1]:
                    if 'label' in graph.nodes[first_port[index]]:
                        new_label = self._get_new_label(node[1]['label'], graph.nodes[first_port[index]]['label'])
                        graph.nodes[first_port[index]]['label'] = new_label
                    else:
                        graph.nodes[first_port[index]]['label'] = node[1]['label']

        # Set the mapping for the pattern port node
        for p in ports:
            mapping[p[1]] = p[0]

        # Create pattern edges with their labels
        for edge in pattern.edges(data=True):
            if edge[0] in second_port:
                port = first_port[second_port.index(edge[0])]
                graph.add_edge(port, mapping[edge[1]], label=edge[2]['label'])
            elif edge[1] in second_port:
                port = first_port[second_port.index(edge[1])]
                graph.add_edge(mapping[edge[0]], port, label=edge[2]['label'])
            elif edge[0] == edge[1] and edge[0] in port[1]:
                port = first_port[second_port.index(edge[1])]
                graph.add_edge(port[0], port, label=edge[2]['label'])
            else:
                graph.add_edge(mapping[edge[0]], mapping[edge[1]], label=edge[2]['label'])

    def _get_new_label(self, first_label, second_label):
        """ Associate two given list of label
        Parameters
        -----------
        first_label
        second_label
        Returns
        -------
        list
        """
        if type(second_label) is str:
            if type(first_label) is str:
                if first_label != second_label:
                    return first_label, second_label
                else:
                    return second_label
            else:
                label = list()
                label.append(second_label)
                for l in first_label:
                    if l not in label:
                        label.append(l)
                return label
        else:
            label = list(second_label)
            if type(first_label) is str:
                if first_label not in second_label:
                    label.append(first_label)
            else:
                for l in first_label:
                    if l not in label:
                        label.append(l)
            return label

    def _is_ports_equals(self, ports):
        """ Check if the candidate ports list are similar to a given ports list
        Parameters
        ----------
        ports
        Returns
        -------
        bool"""
        if len(ports) != len(self.port):
            return False
        else:
            return not (False in [p in ports for p in self.port])

    def compute_description_length(self, label_codes):
        """ Compute description length from the label codes to the candidate merge pattern
        Parameters
        ----------
        label_codes
        """
        if self.final_pattern is not None and label_codes is not None:
            self.code_length = label_codes.encode(self.final_pattern())
        else:
            raise ValueError("You should create the final pattern and set the label codes before computing")

    def inverse(self):
        """ Provide the candidate inverse
        Returns
        -------
        Candidate
        """
        ports = []
        for p in self.port:
            ports.append((p[1], p[0]))
        c = self.first_pattern_label
        self.first_pattern_label = self.second_pattern_label
        self.second_pattern_label = c
        self.port = ports
        del c

        return self

    def __str__(self) -> str:
        return "<{},{},{}>".format(self.first_pattern_label, self.second_pattern_label, self.port)

    def __eq__(self, o: object) -> bool:
        return o.first_pattern_label == self.first_pattern_label \
               and o.second_pattern_label == self.second_pattern_label \
               and self._is_ports_equals(o.port)
