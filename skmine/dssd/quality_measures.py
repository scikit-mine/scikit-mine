from abc import ABC, abstractmethod
import math
from typing import Any, Callable, Dict, List
from pandas import DataFrame, Series
from tslearn.metrics import dtw
from tslearn.barycenters import dtw_barycenter_averaging as dba
from tslearn.barycenters import euclidean_barycenter as eub
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from .utils import column_shares
from .custom_types import ColumnShares

def ones_fraction(df: DataFrame, target_attr: str):
    """Compute the fraction of ones or true value for the target attribute in the dataset"""
    if len(df) == 0:
        raise ValueError("The dataframe can not be empty")

    return len(df[(df[target_attr] == 1) | (df[target_attr] == True)]) / len(df)


def wracc(dataset: DataFrame, candidate_df: DataFrame, target_attr: str):
    """Compute the weighted relative accuracy for """
    dataset_ones = ones_fraction(dataset, target_attr)
    candidate_ones = ones_fraction(candidate_df, target_attr)
    result = (len(candidate_df) / len(dataset)) * abs(candidate_ones - dataset_ones)
    print(f"COMPUTING WRACC FOR target_attr={target_attr}, result={result}")
    return result


def smart_kl(p: Dict[Any, float], q: Dict[Any, float]):
    """Compute the Kullback and Lieber difference as proposed in the paper

    Args:
        s (Series): The series of values of the target attribute that is inherently used by the two probability distributions
        p (Dict[Any, float]): Correct probability distribution 
        q (Dict[Any, float]): Wrong probability distribution

    Returns:
        float
    """
    result = 0.
    # print("INSIDE KL COMPUTING")
    for x in q:
        # print(f"x={x}, p[x]={p[x]}, q[x]={q[x]}")
        if (p[x] != 0 and q[x] != 0):
            result += p[x] * math.log2(p[x] / q[x])

    return result


def smart_kl_sums(entire_df_column_shares: ColumnShares, candidate_df_column_shares: ColumnShares, model_attributes: List[str]):
    """again simple translation of the formula but tried to use the create hat function in order to keep the whole function as similar as possible to the code presented in the paper"""

    kl_sums = 0
    for mi in model_attributes:
        # print(f"COMPUTING KL_SUMS FOR ATTRIBUTE {mi}")
        # print(f"{mi=candidate_df_column_shares[mi=} {mi=entire_df_column_shares[mi=}")
        kl_sums += smart_kl(
            p=candidate_df_column_shares[mi], 
            q=entire_df_column_shares[mi]
        )

    return kl_sums
    # print(f"COMPUTING WKL FOR model_attributes={model_attributes}, result={result}")


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
    def __init__(self, dataset: DataFrame, model_attributes: List[str]) -> None:
        self.dataset = dataset
        self.model_attributes = model_attributes

    @abstractmethod
    def compute_quality(self, sg: DataFrame) -> float:
        pass


class WRACCQuality(QualityMeasure):
    def __init__(self, dataset: DataFrame, binary_model_attribute: str) -> None:
        super().__init__(dataset, [binary_model_attribute])

    def compute_quality(self, sg: DataFrame):
        return wracc(self.dataset, sg, self.model_attributes[0])


class KLQuality(QualityMeasure):
    def __init__(self, dataset: DataFrame, model_attributes: List[str]) -> None:
        super().__init__(dataset, model_attributes)
        self.df_column_shares =  column_shares(dataset)

    def compute_quality(self, sg: DataFrame):
        res = smart_kl_sums(self.df_column_shares, column_shares(sg, self.model_attributes), self.model_attributes)
        return res


class WKLQuality(KLQuality):
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