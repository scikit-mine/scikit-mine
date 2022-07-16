from collections import defaultdict
from distutils.file_util import write_file
import time
import logging
from typing import Generic, List, TypeVar
from pandas import DataFrame
from skmine.base import BaseMiner
from .refinement_operators import RefinementOperator, RefinementOperatorImpl
from .utils import _min_max_avg_quality_string, sort_subgroups, remove_duplicates, subgroup, subgroup, subgroups_to_csv, dummy_logger
from .subgroup import Subgroup
from .description import Description
from .custom_types import  FuncCover, FuncQuality
from .selection_strategies import Desc, SelectionStrategy
from .quality_measures import QualityMeasure


def apply_dominance_pruning(candidate: Subgroup, quality_func: FuncQuality, cover_func: FuncCover):
    """
    Apply dominance pruning to the candidate which consits of removing 
    conditions which absence doesn't descrease the subgroup quality

    Parameters
    ----------
    candidate: Subgroup
        The subgroup to prune
    quality_func: FuncQuality
        A function to be used to re-compute subgroup quality while some 
        of its conditions are being removed 
    cover_func: FuncCover
        A function to be used to re-compute subgroup cover in order 
        to have accurate cover while computing quality
    
    References
    ----------
    [1] Page 226
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    
    # consider the conditions in each subgroup one by one and if removing a condition from a subgroup does not decrease its quality remove that condition permanently
    highest_quality = candidate.quality
    highest_cover = candidate.cover
    for (i, condition) in enumerate(candidate.description.conditions.copy()):
        # remove the condition from the candidate
        candidate.description.conditions.remove(condition)

        candidate.cover = cover_func(candidate).index
        new_quality = quality_func(candidate)
        if new_quality < highest_quality:
            # Add conditions back cause removing this condition decreased the quality of the candidate
            candidate.description.conditions.insert(i, condition)
            candidate.cover = highest_cover
        else: # removing this condition doesn't negatively affect quality so permanently remove it and update the candidate quality to the newly computed one
            highest_quality = new_quality
            candidate.quality = new_quality
            highest_cover = candidate.cover
    

def update_topk(result: List[Subgroup], candidate: Subgroup, max_size: int = 0):
    """
    Insert the new candidate to keep the result list sorted descending with at most the maximum specified size

    Parameters
    ----------
    result: List[Subgroup]
        The destination list where the candidate is to be inserted
    candidate: Subgroup
        The candidate to be inserted
    max_size: int, default=0
        The maximum size allowed for the result list

    Returns
    -------
    List[Subgroup]:
        The same list received in argument is returned for convenience purposes
    """
    
    # highly inspired by the bisect insort method in python 3.8
    def __insort(a: List, x, lo=0, hi=None, key=None):
        hi = hi if hi is not None else len(result)
        _x = key(x)
        while lo < hi:
            mid = (lo + hi) // 2
            if _x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
        a.insert(lo, x)

    # Insert the candidate into the result list in a way to keep the result list sorted descending
    __insort(result, candidate, key= lambda c: -c.quality)
    
    if len(result) > max_size:
        result.pop()
    return result


def mine(
    k: int,
    j: int,
    max_depth: int = 3,
    beam_width: int = None,
    quality: QualityMeasure = None,
    selector: SelectionStrategy = Desc(),
    post_selector: SelectionStrategy = None,
    ref_op: RefinementOperator = RefinementOperatorImpl(),
    output_folder: str = "",
    save_intermediate_results: bool = True,
    save_result: bool = False,
    save_params: bool = False,
    skip_phase2: bool = False,
    skip_phase3: bool = False,
    return_pool: bool = False,
    sort_final_result: bool = False,
    logger: logging.Logger = dummy_logger) -> List[Subgroup]:
    """
    Mine and return mined subgroups

    Parameters
    ----------
    k: int
        The number of subgroups to be returned as a result of the experiment
    j: int 
        The maximum number of subgroups to be kept in memory at all time during the process
    max_depth: int, default=3
        Maximum number of conditions per subgroup description
    beam_width: int, default=k
        The beam width to use during phase 1
    quality: QualityMeasure, default=None
        An implementation of a quality measure to be used. The quality measure uses a projection of the dataset on the target attributes.
    selector: SelectionStrategy, default=Desc()
        An implementation of a selection strategy to be used (during phase 1).
    post_selector: SelectionStrategy, default=selector
        An implementation of a selection strategy to be used (during phase 3).
    ref_op: RefinementOperator, default=RefinementOperatorImpl()
        An implementation of the refinement operator to be used. 
        Notes: The operator is expected to operate on a projection of the dataset on the descriptive attributes, so this is where the descriptive attributes go and minimum coverage, min quality, numeric discretisation cut points variables are exploited.
        (The quality and cover computing function are automatically filled in later by the algorithm)
    output_folder: str, default=""
        The output folder where to store (intermediate) results and any other files created by the algorithm
    save_intermediate_results: bool, default=True
        whether or not to save intermediate results at each depth of the search phase
    save_result: bool, default=False
        whether or not to save the actual result of the mining process
    save_params: bool, default=False
        whether or not to save the parameters used to call this function (might be useful to later identify parameters that produced a specific result)
    skip_phase2: bool, default=False
        Whether or not to skip phase 2
    skip_phase3: bool, default=False
        Whether or not to skip phase 3
    sort_final_result: bool, default=False
        Whether or not to sort the candidates quality descending during phase 3 after post selection.
    logger: Logger, default=dummy_logger
        Logger for the dssd algorithm and other parts

    Returns
    -------
    List[Subgroup] | (Tuple[List[Subgroup], List[Subgroup]]):
        Return the actual result and the pool used for final seleciton if return_pool is True 

    References
    ----------
    [1] Page 224-225
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    # write a string version of all the arguments received to a config file, helpful to later remember what config yielded what result
    beam_width = beam_width if beam_width is not None else k
    if save_params: write_file(f"{output_folder}/dssd_params.conf", [f'{k}={v}' for k,v in locals().items() if k != "logger"])

    ref_op.cover_func = lambda c: subgroup(ref_op.df.loc[c.parent.cover], c.description, True)
    ref_op.quality_func = quality.quality_from_subgroup
    beam = [Subgroup(Description([]), 0, cover=ref_op.df.index, parent=None)] # base subgroup covering the entire original dataset
    pool: List[Subgroup] = []
    result: List[Subgroup] = []
    depth = 1

    start_time = time.time()
    logger.info(f"Phase 1: Mining {j} subgroups each having at most {max_depth} condition(s) each")
    while depth <= max_depth:
        candidates: List[Subgroup] = []
        logger.info(f"depth = {depth}")
        for cand in beam:
            candidates += ref_op.refine_candidate(cand)
        logger.info(f"Generated {len(candidates)} candidates, {_min_max_avg_quality_string(candidates, ' ')}")

        if len(candidates) == 0:
            logger.warning(f"No more valid candidates: stopping exploration phase\n")
            break

        for cand in candidates:
            update_topk(pool, cand, j)
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-generated-candidates.csv", [subgroups_to_csv(candidates)])

        beam = selector.select(candidates, beam_width=beam_width)
        logger.info(f"selected {len(beam)} candidates\n")
        if save_intermediate_results: write_file(f"{output_folder}/depth{depth}-selected-candidates.csv", [subgroups_to_csv(beam)])
        depth += 1
    if save_intermediate_results: write_file(f"{output_folder}/stats1-after-mining-and-before-pruning.csv", [subgroups_to_csv(pool)])

    # post processing phase
    if not skip_phase2:
        logger.info(f"Phase 2: Post processing {len(pool)} candidates")
        logger.info(f"Dominance pruning...")
        for cand in pool:
            apply_dominance_pruning(cand, quality.quality_from_subgroup, lambda c: subgroup(ref_op.df, c.description, False))

        # remove duplicates (with respect to subgroup description) that may have occured due to the pruning process
        logger.info(f"Removing duplicates...")
        pool = remove_duplicates(pool)
        logger.info(f"{len(pool)} remaining after candidates deduplication...")
        if save_intermediate_results: write_file(f"{output_folder}/stats2-after-duplicates-removal.csv", [subgroups_to_csv(pool)])

    # final candidates selection / post selection phase
    if not skip_phase3:
        logger.info(f"Phase 3: Post selecting {k} candidates out of {len(pool)}...")
        result = post_selector.select(pool, beam_width=k)
        if sort_final_result: sort_subgroups(result)

    logger.info(f"Total time taken = {time.time() - start_time} seconds\n")
    logger.info(f"Final selection\n{len(result)} candidate(s)\n{_min_max_avg_quality_string(result, ' ')}")
    if save_result: write_file(f"{output_folder}/stats3-final-results.csv", [subgroups_to_csv(result)])

    return (result, pool) if return_pool else result


def setup_logging(logger_name: str, output_folder: str, level = logging.DEBUG):
    """
    Setup a logger to be used by the dssd algorithm (and components).

    Parameters
    ----------
    logger_name: str
        The name to give for the logger
    output_folder: str
        The output folder to use for logging handler that output their logs to a file
    level: int, default=logging.DEBUG
        The initial loglevel to set for the "new" logger

    Returns
    -------
    logging.Logger:
        A logger with the specified name and handlers to output logs to specified output folder
    """

    logger = logging.getLogger(logger_name)

    if logger.hasHandlers(): 
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    file_debug_handler = logging.FileHandler(f"{output_folder}/debug.log")
    file_info_handler = logging.FileHandler(f"{output_folder}/info.log")
    c_handler.setLevel(logging.DEBUG)
    file_debug_handler.setLevel(logging.DEBUG)
    file_info_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_format = logging.Formatter('%(asctime)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s: %(message)s')
    c_handler.setFormatter(c_format)
    file_info_handler.setFormatter(f_format)
    file_debug_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(file_info_handler)
    logger.addHandler(file_debug_handler)

    logger.setLevel(level)

    return logger


R = TypeVar('R', bound=RefinementOperator)
Q = TypeVar('Q', bound=QualityMeasure)
S = TypeVar('S', bound=SelectionStrategy)
class DSSDMiner(BaseMiner, Generic[Q, S]):
    """
    A scikit mine interface to the dssd algorithm

    Parameters
    ----------
    k: int
        The number of subgroups to be returned as a result of the experiment
    quality: QualityMeasure
        A proper implementation of a quality measure suited for handling the target attributes to be used.
        (The quality measure is expected to operate only a projection of the dataset on the target attributes)
    min_cov: int, default=2
        The minimum coverage for supgroups to be considered
    j: int, default=k*k
        The maximum number of subgroups to be kept in memory at all time during the process.
    max_depth: int, default=3
        Maximum number of conditions per subgroup description
    selector: SelectionStrategy, default=Desc()
        An implementation of a selection strategy to be used.
        During instance creating, the post_selector is also assigned this selector. This can later be manually customized

    Attributes
    ----------
    beam_width: int, default=k
        The beam width to use during DSSd phase 1
    post_selector: SelectionStrategy, default=selector
        An implementation of a selection strategy to be used (during phase 3).
    refinement_operator: RefinementOperator, default=RefinementOperatorImpl()
        An implementation of the refinement operator to be used.
        The operator is expected to operate on a projection of the dataset on the descriptive attributes, therefore it should be able to handle the types of the descriptive attributes.
        The quality and cover computing function are automatically filled in later during the execution of the dssd algorithm so that the operator always perform correclty when actually being used.
    output_folder: str, default=""
        The output folder where to store (intermediate) results, or execution arguments
    save_intermediate_results: bool, default=True
        whether or not to save intermediate results at each depth of phase 1 and also intermediate results at the pruning and deduplication phase (phase 2)
    save_result: bool, default=False
        whether or not to save the actual result of the mining process
    save_params: bool, default=False
        whether or not to save the parameters used to call this function (might be useful to later identify parameters that produced a specific result)
    skip_phase2: bool, default=False
        Whether or not to skip phase 2. Please do not change this variable's default value unless you really know what you are doing
    skip_phase3: bool, default=False
        Whether or not to skip phase 3. Please do not change this variable's default value unless you really know what you are doing
    result: List[Subgroup], default=[]
        This stores the subgroups actually mined after a call to fit(...)
    pool: List[Subgroup], default=[]
        This stores a copy of the final pool of subgroups from which the self.results were selected. It might be intersting to manually inspect this variable and call different selection strategies on this pool to have a different result instead of having to through the entire fitting process again.
        This will likel have a size around self.j
    sort_final_result: bool, default=False
        Whether or not to sort the candidate by during phase 3 after post selection.
        This is set to False so that the order to preserve the selection order as that can be an important information to identify what subgroups were selected first (achieved higher diversity).
    logger: Logger, default=dummy_logger
        This logger is drilled down to all the components of the dssd algorithm. So it can be customized to either output nothing, or be bery verbose or you can pass in your own and it will be used.
        The methods `self.use_*_logger()` modify this attribute with predefined values.

    References
    ----------
    [1] Page 224-225
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, k: int, quality: Q, min_cov: int = 2, j: int = None, max_depth: int = 3, selector: S = Desc()) -> None:
        super().__init__()
        self.k = k
        self.j = j if j is not None else k*k
        self.max_depth = max_depth
        self.quality : Q = quality # TODO set a None default value to this and have a utility function pick a default quality measure later according to the model attributes
        self.selector = selector # TODO set a None default value and have a function automatically choose the the potentially best selection strategy depending on the data at hand
        self.post_selector = selector
        self.refinement_operator: R = RefinementOperatorImpl(min_cov=min_cov)
        self.beam_width = k
        self.output_folder = ""
        self.save_intermediate_results = False
        self.save_result = False
        self.save_params = False
        self.skip_phase2 = False
        self.skip_phase3 = False
        self.result: List[Subgroup] = []
        self.pool: List[Subgroup] = []
        self.logger = dummy_logger
        self.sort_final_result = False


    def fit(self, D: DataFrame, y: DataFrame):
        """
        Extract self.k subgroups by describing them using from the first 
        dataframe and weighing their quality based on the second dataframe.

        Parameters
        ----------
        D: DataFrame
            dataframe containing the descriptive attributes/columns
        y: Dataframe
            dataframe containing the target/model attributes/columns

        Returns
        -------
        DSSDMiner

        Examples
        --------
        >>> from skmine.datasets.dssd import load_emotions # doctest: +SKIP
        >>> from skmine.dssd import DSSDMiner, WKLQuality, VarCoverBasedSelectionStrategy # doctest: +SKIP
        >>> D,y = fetch_emotions(return_D_y = True) # doctest: +SKIP
        >>> dssd = DSSDMiner(k=100, j=10_000, min_cov=10, max_depth=3, quality_measure=WKLQuality(), selector=VarCoverBasedSelectionStrategy) # doctest: +SKIP
        >>> dssd.fit(df[descriptive_attributes], df[model_attributes]).discover() # doctest: +SKIP
        >>> dssd.result # doctest: +SKIP
        """
        self.refinement_operator.df = D
        self.quality.df = y
        (self.result, self.pool) = mine(
            k=self.k, 
            j=self.j,
            max_depth=self.max_depth,
            beam_width=self.beam_width,
            quality=self.quality,
            selector=self.selector,
            post_selector=self.post_selector,
            ref_op=self.refinement_operator,
            output_folder=self.output_folder,
            save_intermediate_results=self.save_intermediate_results,
            save_result=self.save_result,
            save_params=self.save_params,
            skip_phase2=self.skip_phase2,
            skip_phase3=self.skip_phase3,
            sort_final_result=self.sort_final_result,
            return_pool=True,
            logger=self.logger
        )
        return self


    def _discover(self, result: List[Subgroup], return_cover: bool = False):
        data = defaultdict(list)
        for c in result:
            data["quality"].append(c.quality)
            data["pattern"].append(c.description)
            data["pattern_length"].append(len(c.description))
            data["cover_length"].append(len(c))
            data["cover"].append(list(c.cover))
        if not return_cover:
            del data["cover"]
        return DataFrame(data=data)


    def discover(self, return_cover: bool = False):
        """
        Navigate through the result of a fitting process

        Parameters
        ----------
        return_cover: bool, default=False
            Whether or not return a column containing the actual content of each subgroup selected 

        Returns
        -------
        DataFrame:
            A dataframe containing the following columns (quality, pattern, pattern_length, cover_length) and optionnaly the cover column depending on the return_cover parameter
        """
        return self._discover(self.result, return_cover)


    def discover_pool(self, return_cover: bool = False):
        """
        Navigate through the pool of a fitting process

        Parameters
        ----------
        return_cover: bool, default=False
            Whether or not return a column containing the actual content of each subgroup from the final pool 

        Returns
        -------
        DataFrame:
            A dataframe containing the following columns (quality, pattern, pattern_length, cover_length) and optionnaly the cover column depending on the return_cover parameter
        """
        return self._discover(self.pool, return_cover)


    def use_verbose_logger(self, output_folder: str):
        """
        Use a preconfigured verbose logger that enables in depth inspection during the fitting process.
        This method also enables saving intermediate and final results.

        Parameters
        ----------
        output_folder: str
            An existing folder to store all the verbose logging, the intermediate and final results of a fitting process

        Returns
        -------
        DSSDMiner:
            self
        """
        self.output_folder = output_folder
        self.save_params = self.save_result = self.save_intermediate_results = True
        self.logger = setup_logging("dssd_miner", self.output_folder)
        return self


    def use_silent_logger(self):
        """
        Use a dummy logger that silences any form of output during the fitting process.
        This method also disables saving intermediate and final results.

        Returns
        -------
        DSSDMiner:
            self
        """
        self.output_folder = ""
        self.save_params = self.save_result = self.save_intermediate_results = False
        self.logger = dummy_logger
        return self


    def use_selector(self, selector: SelectionStrategy):
        """
        Updates selection strategies to be used during next fitting process.

        Parameters
        ----------
        selector: SelectionStrategy
            An implementation of a selection strategy to be used during both phase1 and phase 3 of the algorithm.

        Returns
        -------
        DSSDMiner:
            self
        """
        self.post_selector = self.selector = selector
        return self


    def _deduplicate_cover(self, data: List[Subgroup]):
        tmp_dict = {}
        for s in data:
            hash_value = hash(tuple(set(s.cover)))
            if hash_value not in tmp_dict:
                tmp_dict[hash_value] = s
        return list(tmp_dict.values())


    def deduplicate_pool(self):
        """
        Remove any duplicates in `self.pool`  with respect to subgroups' cover

        Returns
        -------
        DSSDMiner:
            self
        """
        self.pool = self._deduplicate_cover(self.pool)
        return self


    def deduplicate_result(self):
        """
        Remove any duplicates in `self.result`  with respect to subgroups' cover

        Returns
        -------
        DSSDMiner:
            self
        """
        self.result = self._deduplicate_cover(self.result)
        return self
