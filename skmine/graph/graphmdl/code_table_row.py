from skmine.graph.graphmdl import utils
import math


class CodeTableRow:
    """
        Object to represent a row of the code table
        Its concerns only non-singleton pattern

        Parameters
        ----------
        pattern
            It's the row pattern structure
        pattern_usage:int ,default=None
            It's the row usage after a cover
        pattern_port_usage: dict ,default=None
            This is the usage of each port of the line pattern after the cover.
    """

    def __init__(self, pattern, pattern_usage=None, pattern_port_usage=None):
        self._pattern = pattern
        self._pattern_usage = pattern_usage
        self._pattern_port_usage = pattern_port_usage
        self._embeddings = []  # the pattern embeddings in the data
        self._code_length = 0.0
        self._port_code_length = None  # the pattern ports code
        self._description_length = 0.0
        self._used_embeddings = []  # used embeddings
        self.arrival_number = None

    def code_length(self):
        """ Provide the row code length
        Returns
        -------
        float
        """
        return self._code_length

    def port_code_length(self):
        """ Provide the row ports code length
        Returns
        -------
        dict
        """
        return self._port_code_length

    def pattern(self):
        """ Provide the row pattern
        Returns
        -------
        object
        """
        return self._pattern

    def pattern_usage(self):
        """ Provide the row pattern code
        Returns
        -------
        float
        """
        return self._pattern_usage

    def set_pattern_usage(self, pattern_code):
        """ Set the row pattern code
        Parameters
        ----------
        pattern_code
        """
        self._pattern_usage = pattern_code

    def pattern_port_usage(self):
        """ Provide the port code of the row pattern
        Returns
        -------
        dict
        """
        return self._pattern_port_usage

    def set_pattern_port_usage(self, port_code):
        """ set the port code of the row pattern
        Parameters
        ---------
        port_code
        """
        self._pattern_port_usage = port_code

    def set_embeddings(self, embeddings):
        """ Set the pattern row embeddings
        Parameters
        ----------
        embeddings
        """
        self._embeddings = embeddings

    def add_used_embeddings(self, embedding):
        self._used_embeddings.append(embedding)

    def used_embeddings(self):
        return self._used_embeddings

    def embeddings(self):
        """ Provide the pattern row embeddings
        Returns
        -------
        list
        """
        return self._embeddings

    def compute_code_length(self, rows_usage_sum):
        """ Compute the code length of the row and its ports
        Parameters
        ---------
        rows_usage_sum : total of usage for the code table rows
        """
        self._port_code_length = dict()

        # Compute pattern code length
        if self._pattern_usage == 0:
            self._code_length = 0.0
        else:
            self._code_length = utils.log2(self._pattern_usage, rows_usage_sum)

        # compute port usage sum
        port_usage_sum = 0.0
        for k in self._pattern_port_usage.keys():
            port_usage_sum = port_usage_sum + self._pattern_port_usage[k]

        # compute each port code length
        for p in self._pattern_port_usage.keys():
            if self._pattern_port_usage[p] == 0:
                self._port_code_length[p] = 0.0
            else:
                code = utils.log2(self._pattern_port_usage[p], port_usage_sum)
                if code == - 0.0:
                    self._port_code_length[p] = 0.0
                else:
                    self._port_code_length[p] = code

    def compute_description_length(self, standard_table):
        """ Compute the row  description length according kgmdl equation
        Parameters
        ---------
        standard_table
        """
        if self._pattern_usage is None:
            self._description_length = 0.0

        if self._pattern_port_usage is None or self._port_code_length is None:
            raise ValueError("Row's codes should be compute")
        self._description_length = 0.0

        code_port_total = 0.0
        for value in self._port_code_length.values():
            code_port_total += value

        port_desc = math.log2(len(self._pattern.nodes()) + 1)
        port_desc += math.log2(utils.binomial(len(self._pattern.nodes()), len(self._port_code_length)))
        # port_desc += code_port_total

        # self._description_length = self._code_length  # usage description
        self._description_length += utils.encode(self._pattern, standard_table)  # structure description
        self._description_length += port_desc  # ports description

    def description_length(self):
        """ Provide the row description length
        Returns
        -------
        float
        """
        return self._description_length

    def display_row(self):
        """ Display a row in a certain format
        Returns
        -------
        list
        """
        return [utils.draw_pattern(self._pattern), self._pattern_usage, self._code_length,
                len(self._pattern_port_usage),
                self._pattern_port_usage, self._port_code_length]

    def __str__(self):
        return "{} | {} |{} |{} |{} |{}" \
            .format(utils.display_graph(self._pattern), self._pattern_usage, self._code_length,
                    len(self._pattern_port_usage), self._pattern_port_usage,
                    self._port_code_length)
