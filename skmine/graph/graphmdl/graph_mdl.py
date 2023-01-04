import copy
import logging
import time
import networkx
from skmine.base import BaseMiner
from skmine.graph.graphmdl.label_codes import LabelCodes
from skmine.graph.graphmdl.code_table import CodeTable
from skmine.graph.graphmdl.code_table_row import CodeTableRow
from skmine.graph.graphmdl import utils


def _order_pruning_rows(row):
    """ Sort the pruning row by their usage"""
    return row.pattern_usage()


class GraphMDL(BaseMiner):
    """
        This is a python re-implementation of the GraphMDL family of approaches for extracting a small and
        descriptive set of graph patterns from graph data.
        This re-implementation supports directed graphs only, but supports multi-graphs.

        This is a python re-implementation of the original Java algorithm, which is available at 'https://gitlab.inria.fr/fbariatt/graphmdl'.
        As such, some functionalities are not available in this version, such as a full support of graph automorphisms.

        Author: Arnauld Djedjemel

        References
        ----------
         F. Bariatti, "Mining Tractable Sets of Graph Patterns with the Minimum Description Length Principle",
         PhD thesis, Université de Rennes 1, 2021. Available: https://hal.inria.fr/tel-03523742

         F. Bariatti, P. Cellier, and S. Ferré. "GraphMDL+ : interleaving the generation and MDL-based selection of
         graph patterns", in Proceedings of the 36th Annual ACM Symposium on Applied Computing, Mar. 2021, pp. 355–363.
         doi: 10.1145/3412841.3441917.
    """

    def __init__(self):
        self._data = None
        self._label_codes = None
        self._code_table = None
        self._rewritten_graph = None
        self._description_length = 0.0
        self._patterns = set()
        self._already_test = []  # list of the candidates already tested
        self._pruning_rows = []  # list of the rows who can prune
        self._old_usage = dict()  # Mapper for the rows old usage, useful for the pruning
        self._timeout = None
        self._initial_description_length = 0

    def _init_graph_mdl(self):
        """
            Initialize the GraphMDL elements such as the label code,
            the initial code table (CT0), and run a cover computation to create the first rewritten graph.
        """
        self._already_test = []
        self._label_codes = LabelCodes(self._data)  # label codes creation
        # CT0 creation
        self._code_table = CodeTable(self._label_codes, self._data)
        # CT0 cover
        self._code_table.cover()
        self._rewritten_graph = self._code_table.rewritten_graph()
        self._description_length = self._code_table.compute_total_description_length()
        self._initial_description_length = self._description_length
        utils.MyLogger.info(f"\n initial CT \n {self._code_table}")
        utils.MyLogger.info("GraphMDL+ run ...")
        utils.MyLogger.info(f"Initial description length = {round(self._description_length, 2)}")

    def fit(self, D, timeout=None):
        """
            Execute GraphMDl on a given data graph

            Parameters
            ----------
            D : networkx graph. All edges need to be labeled.
            timeout: int , default=None
                Maximum time for the algorithm execution (approx.).

            Returns
            -------
            GraphMDL
        """
        if timeout is not None and timeout != 0:
            self._timeout = timeout
        if D is None:
            raise ValueError("You should give a graph")
        else:
            self._data = D
            self._init_graph_mdl()
            self._anytime_graph_mdl_with_timeout()
            return self

    def _anytime_graph_mdl_with_timeout(self):
        """ Anytime graph mdl with timeout
        Returns
        --------
        GraphMDL
        """
        begin = time.time()
        current = 0
        stop = False
        while not stop:
            if self._timeout is not None:
                if current > self._timeout:
                    stop = True
                    break
            utils.MyLogger.info("Candidate generation and sort start .....")
            b = time.time()
            candidates = utils.generate_candidates(self._rewritten_graph, self._code_table)
            candidates.sort(reverse=True, key=self._order_candidates)
            utils.MyLogger.info(f"Candidate generation and sort end ..........time ={time.time() - b}")
            utils.MyLogger.info(f"candidates number {len(candidates)}")
            utils.MyLogger.info("GraphMDL best Ct search start .....")
            if self._timeout is not None:
                current = time.time() - begin
                if self._stop_by_time(current, self._timeout):
                    break
            if len(candidates) != 0:
                b = time.time()
                for candidate in candidates:
                    if candidate not in self._already_test:
                        # Add a candidate to a ct, cover and compute description length
                        if self._timeout is not None:
                            current = time.time() - begin
                            if self._stop_by_time(current, self._timeout):
                                break
                        self._compute_old_usage()
                        row = CodeTableRow(candidate.final_pattern())
                        self._code_table.add_row(row)
                        self._code_table.cover()
                        temp_code_length = self._code_table.compute_total_description_length()
                        self._already_test.append(candidate)
                        # if the new ct is better than the old, break and generate new candidates
                        # with the new ct
                        if temp_code_length < self._description_length:
                            self._rewritten_graph = self._code_table.rewritten_graph()
                            self._description_length = temp_code_length
                            utils.MyLogger.info("New DL ", self._description_length)
                            utils.MyLogger.info(f"new pattern added: {utils.display_graph(row.pattern())}")
                            utils.MyLogger.info(f"search time = {time.time() - b}")
                            self._compute_pruning_candidates()
                            self._pruning()
                            break
                        elif temp_code_length > self._description_length and candidates.index(candidate) == len(
                                candidates) - 1:
                            utils.MyLogger.info("None best code table found")
                            utils.MyLogger.info(f"search time = {time.time() - b}")
                            self._code_table.remove_row(row)
                            stop = self._graph_mdl_end()
                            if self._timeout is not None:
                                current = self._timeout
                            break
                        else:
                            # if the candidate not improve the result, remove it to the code table
                            self._code_table.remove_row(row)
                    else:
                        utils.MyLogger().debug("Already test")
            else:
                stop = self._graph_mdl_end()
                if self._timeout is not None:
                    current = self._timeout
        return self

    def _stop_by_time(self, passed_time, timeout):
        """
            Check if the passed time surpasses the timeout

            Parameters
            ---------
            passed_time : int
                The passed time
            timeout : int
                The maximum time

            Returns
            --------
            bool
        """
        if passed_time >= timeout:
            return self._graph_mdl_end()
        else:
            return False

    def _graph_mdl_end(self):
        """
            Complete the graph mdl algorithm by cover the code table and stop

            Returns
            -------
            bool
        """
        utils.MyLogger.info("GraphMDL+ end .....")
        self._code_table.cover()
        self._rewritten_graph = self._code_table.rewritten_graph()
        self._description_length = self._code_table.compute_total_description_length()
        utils.MyLogger.info(f"Final description length = {round(self._description_length, 2)}")
        utils.MyLogger.info(f"Number of patterns found = {len(self.patterns())}")

        return True

    def _order_candidates(self, candidate):
        """
            Provide the candidate elements to order candidates
            Parameters
            ----------
            candidate

            Returns
            -------
            list
        """
        return [candidate.usage, candidate.exclusive_port_number,
                -self._label_codes.encode(candidate.final_pattern())]

    def _compute_old_usage(self):
        """
            Store patterns usage before compute new code table patterns usage
        """
        self._old_usage = dict()
        for r in self._code_table.rows():
            self._old_usage[r.pattern()] = r.pattern_usage()

    def _compute_pruning_candidates(self):
        """
            Find the row where their usage has decreased since the last usage,
            it's the step before the code table pruning
        """
        for r in self._code_table.rows():
            if r.pattern() in self._old_usage.keys():
                if r.pattern_usage() < self._old_usage[r.pattern()]:
                    self._pruning_rows.append(r)

    def _pruning(self):
        """
            Make the code table pruning as krimp pruning.

            That's consist of remove row in code table who are unnecessary,
            because without them the code table is better.
        """
        utils.MyLogger.info(f"Pruning start ....")
        self._compute_old_usage()  # compute old pattern usage
        self._pruning_rows.sort(key=_order_pruning_rows)  # sort the pruning rows
        for r in self._pruning_rows:
            self._code_table.remove_row(r)
            self._code_table.cover()
            if self._code_table.compute_total_description_length() < self._description_length:
                self._rewritten_graph = self._code_table.rewritten_graph()
                self._description_length = self._code_table.compute_total_description_length()
                self._pruning_rows.remove(r)
                self._compute_pruning_candidates()  # recompute the pruning candidates
                self._pruning()
            else:
                self._code_table.add_row(r)

        utils.MyLogger.info("Pruning end ....")

    def patterns(self):
        """
            Return the patterns found by the algorithm after fit has been called.

            Returns
            -------
            set
        """
        self._patterns = set()
        if self._code_table is not None:
            for r in self._code_table.rows():
                if r.code_length() != 0:
                    self._patterns.add(r.pattern())

            for s in self._code_table.singleton_code_length().keys():
                self._patterns.add(utils.create_singleton_pattern(s, self._code_table))

            return self._patterns

        else:
            return ValueError("The fit method must be called first")

    def description_length(self):
        """
            Return the MDL description length (model + encoded data) for the best code table found by the algorithm.

            Returns
            --------
            float
        """
        return self._description_length

    def initial_description_length(self):
        """
            Return the MDL description length for the initial --singleton-only-- code table CT0.

            Returns
            -------
            float
        """
        return self._initial_description_length

    def discover(self, *args, **kwargs):
        """
            Provide a summary of the algorithm execution
        """
        if self._code_table is not None:
            print(self._code_table.display_ct())
            print("final description length : ", self._description_length)
            print("Non singleton patterns found : ")
            for p in self.patterns():
                print('\n', utils.display_graph(p))
            if len(self._code_table.singleton_code_length()) != 0:
                print("Singleton patterns found : ")
                for s in self._code_table.singleton_code_length().keys():
                    print("\n", s)
        else:
            raise ValueError("The fit method must be called first")

