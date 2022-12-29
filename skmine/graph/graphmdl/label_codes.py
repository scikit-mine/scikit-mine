from networkx import Graph
import skmine.graph.graphmdl.utils as utils


class LabelCodes:
    """
      It is only a storage for label frequency in the initial data graph

      Parameters
      -----------
      graph:Graph
         the treated graph
    """

    def __init__(self, graph: Graph):
        self._total_label = utils.get_total_label(graph)
        self._vertexLC = dict([(u, utils.log2(v, self._total_label))
                               for u, v in utils.count_vertex_label(graph).items()])  # Vertex label code length
        self._edgeLC = dict([(u, utils.log2(v, self._total_label))
                             for u, v in utils.count_edge_label(graph).items()])    # edge label code length

    def display_vertex_lc(self):
        """ Display vertex label code length
            Returns
            --------
            str
       """
        msg = ""
        for i, j in self._vertexLC.items():
            msg += "{}->{}\n".format(i, j)
        return msg

    def display_edge_lc(self):

        """ Display edge label code length

            Returns
            --------
            str
        """
        msg = ""
        for i, j in self._edgeLC.items():
            msg += "{}->{}\n".format(i, j)
        return msg

    def total_label(self):
        """
        Provide the total number of labels
        Returns
        -------
        double
        """
        return self._total_label

    def vertex_lc(self):
        """
        Provide the code length from all vertex label
        Returns
        -------
        dict
        """
        return self._vertexLC

    def edges_lc(self):
        """
        Provide the code length from all edge label
        Returns
        -------
        dict
        """
        return self._edgeLC

    def encode(self, pattern: Graph):
        """ Compute description length of a given pattern with this label codes
        Parameters
        ----------
        pattern
        Returns
        ---------
        float
        """
        return utils.encode(pattern, self)

    def encode_singleton_vertex(self, vertex_singleton_label):
        """ Compute description length of a given vertex singleton pattern with this label codes
        Parameters
        ----------
        vertex_singleton_label
            The given vertex singleton label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 1, vertex_singleton_label)

    def encode_singleton_edge(self, edge_singleton_label):
        """ Compute description length of a given edge singleton pattern with this label codes
        Parameters
        ----------
        edge_singleton_label
        Returns
        ---------
        float
        """
        return utils.encode_singleton(self, 2, edge_singleton_label)

    def __str__(self) -> str:
        return "Edge label\n-----------------\n" + self.display_edge_lc() + "\nVertex label\n----------------------\n" \
               + self.display_vertex_lc()
