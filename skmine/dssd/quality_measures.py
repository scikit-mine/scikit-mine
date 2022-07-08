from abc import ABC, abstractmethod
import math
from typing import Any, Callable, DefaultDict, Dict, List
from pandas import DataFrame
from tslearn.metrics import dtw
from tslearn.barycenters import dtw_barycenter_averaging as dba
from tslearn.barycenters import euclidean_barycenter as eub
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from .utils import column_shares
from .custom_types import ColumnShares


def measure_distance(c1: np.ndarray, c2: np.ndarray, distance_measure: str) -> float:
    """Compute and return a measure of the similarity between two models
        distance_measure = "euclidean", "dtw"
    """

    if distance_measure == "dtw" :  # self > quality measure ??
        # return dtw(c1, c2)
        return fastdtw(c1, c2)[0] # maybe this is slow because it is also computing the path ?
        return fastdtw(c1, c2, dist=euclidean)[0] # maybe this is slow because it is also computing the path ?
    if distance_measure == "euclidean" :
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(c1, c2))) # taken from python official doc to use instead of required to have python 3.8 as this is when the dist function was introduced
        # return math.dist(c1, c2)
        # return math.sqrt(sum((c1 - c2) ** 2))
    
    raise ValueError("!!! UNSUPPORTED DISTANCE MEASURE !!!: Only 'euclidean' and 'dtw' are supported")


def ts_model(subgroup: DataFrame, attr: str, target_model: str = "eub") -> np.ndarray:
    """Compute a model for the subgroup based on the target_model on this instance.
        Model computation is based on the target model technique defined on this instance

    Args:
        subgroup (DataFrame): a df of elements to compute a model for

    Returns:
        numpy.array of shape (sz, d): the computed model for the subgroup
    """
    if target_model == 'dba' :
        return dba(subgroup[attr].to_numpy())
    if target_model == 'eub' :
        return eub(subgroup[attr].to_numpy())
        # model = eub([t for t in subgroup[self.target]])
    raise ValueError("!!! UNSUPPORTED TARGET MODEL !!!: Only 'eub' and 'dba' are supported")



class QualityMeasure(ABC):
    """
    A class representing a method to compute the quality of a subgroup
    based on the target attribute(s).
    """
    def __init__(self, dataset: DataFrame, model_attributes: List[str]) -> None:
        self.dataset = dataset
        self.model_attributes = model_attributes

    @abstractmethod
    def compute_quality(self, sg: DataFrame) -> float:
        """
        Return the quality of the given subgroup

        Parameters
        ----------
        sg: DataFrame
            The cover/content of the subgroup

        Returns
        -------
        float: The actual quality of the subgroup 
        """
        pass


class WRACCQuality(QualityMeasure):
    """
    Compute the weighted relative accuracy quality of a subgroup with regards to a single target binary attribute

    Parameters
    ----------
    dataset: DataFrame
        A dataframe representing the entire dataset
    binary_model_attribute: str
        The name of the target attriute. This attribute should be a binary one 

    References
    ----------
    [1] Page 215-216 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, dataset: DataFrame, binary_model_attribute: str) -> None:
        super().__init__(dataset, [binary_model_attribute])
        self.dataset_ones = self.ones_fraction(dataset, binary_model_attribute)

    @property
    def bin_attr(self) -> str:
        return self.model_attributes[0]

    @classmethod
    def ones_fraction(cls, df: DataFrame, attr: str):
        """Compute the fraction of ones or true values for an attribute in the dataframe"""
        if len(df) == 0:
            raise ValueError("The dataframe can not be empty")
        return len(df[(df[attr] == 1) | (df[attr] == True)]) / len(df)
            
    def compute_quality(self, sg: DataFrame):
        candidate_ones = self.ones_fraction(sg, self.bin_attr)
        result = (len(sg) / len(self.dataset)) * abs(candidate_ones - self.dataset_ones)
        print(f"COMPUTING WRACC FOR target_attr={self.bin_attr}, result={result}")
        return result


class KLQuality(QualityMeasure):
    """
    Compute the Kullback-Leibler quality of a subgroup with regards to
    a single or multiple target binary or nominal attributes.
    Notes: As described in the article from the references, this version does not 
    take into account the size of the subgroup while computing the quality.
    As such, smaller subgroups tend to have higher quality so you might end up 
    with unexpected results if you don't fully understand this measure. It is generaly
    recommended to use the `skmine.dssd.WKLQuality` for more predictable results.

    Parameters
    ----------
    dataset: DataFrame
        A dataframe representing the entire dataset
    model_attributes: List[str]
        The names of the target attributes

    See also
    --------
    skmine.dssd.WKLQuality

    References
    ----------
    [1] Page 217-218 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, dataset: DataFrame, model_attributes: List[str]) -> None:
        super().__init__(dataset, model_attributes)
        self.df_column_shares =  column_shares(dataset)


    @classmethod
    def kl(cls, p: DefaultDict[Any, float], q: DefaultDict[Any, float]) -> float:
        """Compute the Kullback-Leibler difference as proposed in the paper

        Parameters
        ----------
        p: DefaultDict[Any, float]
            Correct probability distribution 
        q: DefaultDict[Any, float]
            Wrong probability distribution

        Returns
        -------
        float

        References
        ----------
        [1] Page 217 (Section on Weighted Kullback–Leibler divergence)
            Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
        """
        result = 0.
        for x in q:
            if (p[x] != 0 and q[x] != 0):
                result += p[x] * math.log2(p[x] / q[x])
        return result


    def compute_quality(self, sg: DataFrame):
        sg_shares = column_shares(sg, self.model_attributes)
        result = 0
        for attr in self.model_attributes:
            result += self.kl(p=sg_shares[attr], q=self.df_column_shares[attr])

        return result


class WKLQuality(KLQuality):
    """
    Compute the Weighted Kullback and Leibler quality of a subgroup with regards to
    a single or multiple target binary or nominal attributes.
    This version is tagged weighted as it takes into account the size of 
    the subgroup while computing the quality, thus gives more predictable results.

    Parameters
    ----------
    dataset: DataFrame
        A dataframe representing the entire dataset
    model_attributes: List[str]
        The names of the target attributes

    References
    ----------
    [1] Page 217-218 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def compute_quality(self, sg: DataFrame):
        return super().compute_quality(sg) * len(sg)


class TSQuality(QualityMeasure):
    def __init__(self, dataset: DataFrame, model_attribute: str, target_model: str, dist_measure: str) -> None:
        super().__init__(dataset, [model_attribute])
        self.target_model = target_model
        self.dist_measure = dist_measure
        self.dataset_model = ts_model(dataset, model_attribute, target_model)


    def compute_quality(self, sg: DataFrame):
        quality = 0
        if not sg.empty:
            dba_c = ts_model(sg, self.model_attributes[0], self.target_model)
            dba_compl = self.dataset_model  # model subgroup compl TODO : si vide ?
            # pow(len(sg),0.5) is used to take the size of the subgroup into account
            # jsigne said this way of doing was decided commonly with the other researchers
            quality = pow(len(sg),0.5) * measure_distance(dba_c.ravel(), dba_compl.ravel(), self.dist_measure)  # *(len(sg)/len(self.data))  # compare model
        return quality



_builders: Dict[str, Callable[..., QualityMeasure]] = None 

if _builders is None:
    _builders = {
        "kl": KLQuality,
        "wkl": WKLQuality,
        "wracc": WRACCQuality,
        "ts_quality": TSQuality,
        "wkg": None
    }

def register(measure_name: str, builder: Callable[..., QualityMeasure]):
    _builders[measure_name] = builder

def create(measure_name: str, entire_df: DataFrame, extra_parameters: Dict[str, Any] ):
    builder = _builders.get(measure_name)
    if not builder:
        raise ValueError(f"!!! UNSUPPORTED QUALITY MEASURE ({measure_name}) !!!")
    return builder(entire_df, **extra_parameters)
