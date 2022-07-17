from abc import ABC, abstractmethod
import math
from typing import Any, DefaultDict
from pandas import DataFrame, Series
import pandas
from tslearn.metrics import dtw
from tslearn.barycenters import dtw_barycenter_averaging as dba
from tslearn.barycenters import euclidean_barycenter as eub
import numpy as np
from fastdtw import fastdtw
from skmine.dssd.subgroup import Subgroup
from .utils import column_shares


class QualityMeasure(ABC):
    """
    A class representing a method to compute the quality of a subgroup
    based on the target attribute(s).

    Parameters
    ----------
    df: DataFrame
        The dataframe containing the projection of the dataset only on target attributes

    Attributes
    ----------
    model_attributes: List[str]
        The list of the model attributes used by this quality measure (this is a get only property)
    """
    def __init__(self, df: DataFrame) -> None:
        self._df = df

    @property
    def model_attributes(self):
        return self._df.columns

    @property
    def df(self): return self._df

    @df.setter
    def df(self, df: DataFrame):
        self._df = df

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
        float: 
            The actual quality of the subgroup 
        """
        pass


    def quality_from_subgroup(self, sg: Subgroup) -> float:
        """
        Return the quality of the given subgroup

        Parameters
        ----------
        sg: Subgroup
            The cover/content of the subgroup

        Returns
        -------
        float:
            The actual quality of the subgroup 
        """
        return self.compute_quality(self.df.loc[sg.cover])

    def quality_from_cover(self, sg: pandas.Index) -> float:
        """
        Return the quality of the given subgroup cover

        Parameters
        ----------
        sg: pandas.Index
            The cover/content of the subgroup

        Returns
        -------
        float:
            The actual quality of the corresponding subgroup 
        """
        return self.compute_quality(self.df.loc[sg])


class WRACC(QualityMeasure):
    """
    Compute the weighted relative accuracy quality of a subgroup with regards to a single target binary attribute

    Parameters
    ----------
    df: DataFrame, default=DataFrame()
        A dataframe representing the entire dataset. The dataframe should only contain one column as WRACC is only meant for single binary target attribute

    Attributes
    ----------
    bin_attr: str
        The name of the binary attribute used by this quality measure (get only property)

    References
    ----------
    [1] Page 215-216 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, df: DataFrame = DataFrame()) -> None:
        super().__init__(df)
        self.df = df # trigger internal update

    def __str__(self) -> str:
        return "wracc"

    @property
    def bin_attr(self) -> str:
        return self.model_attributes[0]

    @QualityMeasure.df.setter
    def df(self, df: DataFrame):
        QualityMeasure.df.fset(self, df)
        self.dataset_ones = self.ones_fraction(df, self.bin_attr)

    @classmethod
    def ones_fraction(cls, df: DataFrame, attr: str):
        """
        Compute the fraction of ones or true values for an attribute in the dataframe
        
        Parameters
        ----------
        df: DataFrame
            The dataframe to use
        attr: str
            The column name to select in the dataframe. This column should only have binary values(bool or 0/1)

        Returns
        -------
        float
        """
        if len(df) == 0:
            raise ValueError("The dataframe can not be empty")
        return len(df[(df[attr] == 1) | (df[attr] == True)]) / len(df)
            
    def compute_quality(self, sg: DataFrame):
        candidate_ones = self.ones_fraction(sg, self.bin_attr)
        result = (len(sg) / len(self._df)) * abs(candidate_ones - self.dataset_ones)
        return result


class KL(QualityMeasure):
    """
    Compute the Kullback-Leibler quality of a subgroup with regards to
    a single or multiple target binary or nominal attributes.
    Notes: As described in the article from the references, this version does not 
    take into account the size of the subgroup while computing the quality.
    As such, smaller subgroups tend to have higher quality so you might end up 
    with unexpected results if you don't fully understand this measure. It is generaly
    recommended to use the `skmine.dssd.WKL` for more predictable results.

    Parameters
    ----------
    df: DataFrame, default=DataFrame()
        A dataframe representing the entire dataset. The dataframe should contain one or multiple binary/nominal columns

    See also
    --------
    WKL

    References
    ----------
    [1] Page 217-218 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, df: DataFrame = DataFrame()) -> None:
        super().__init__(df)
        self.df = df # trigger internal update
    
    def __str__(self) -> str:
        return "kl"

    @QualityMeasure.df.setter
    def df(self, df: DataFrame):
        QualityMeasure.df.fset(self, df)
        self.df_column_shares = column_shares(df)


    @classmethod
    def kl(cls, p: DefaultDict[Any, float], q: DefaultDict[Any, float]) -> float:
        """
        Compute the Kullback-Leibler difference as proposed in the paper

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


class WKL(KL):
    """
    Compute the Weighted Kullback and Leibler quality of a subgroup with regards to
    a single or multiple target binary or nominal attributes.
    This version is tagged weighted as it takes into account the size of 
    the subgroup while computing the quality, thus gives more predictable results.

    Parameters
    ----------
    df: DataFrame, default=DataFrame()
        A dataframe representing the entire dataset. The dataframe should contain one or multiple binary/nominal columns

    References
    ----------
    [1] Page 217-218 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def compute_quality(self, sg: DataFrame):
        return super().compute_quality(sg) * len(sg)

    def __str__(self) -> str:
        return "wkl"



class TSModel(ABC):
    """
    An abstract class representing a method used to compute the average of multiple time series
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def compute_model(self, series: Series) -> np.ndarray:
        """
        Compute an average model for the time series in the specified series

        Parameters
        ----------
        series: Series
            A series containing the time series to compute a model for

        Returns
        -------
        numpy.array of shape (sz, d):
            The computed model for the subgroup
        """
        pass


class TSDistance(ABC):
    """
    An abstract class representing a method used to compute the distance between two time series
    
    Parameters
    ----------
    dist_kwargs:
        A dictionnary of key value arguments to pass to the actual implementation of the computation function 
    """
    def __init__(self, **dist_kwargs) -> None:
        super().__init__()
        self.dist_kwargs = dist_kwargs 
   
    def measure_distance(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """
        Return a distance between two time series (arrays of numerical values)

        Parameters
        ----------
        c1: numpy.ndarray
            The first array of values
        c2: numpy.ndarray
            The other array of values

        Returns
        -------
        float
        """
        pass


class TSQuality(QualityMeasure, TSModel, TSDistance):
    """
    Compute the quality of a subgroup with regards to a single time series attribute
    This method is based on an averaging method and a distance computing method.
    The internal formula used for computing the quality is: `q = sqrt(|subgroup|) * dist(model(dataset), model(subgroup))`

    Parameters
    ----------
    df: DataFrame, default=DataFrame()
        A dataframe representing the entire dataset. The dataframe should contain only one column that contains only time series
    **dist_kwargs:
        Keyword based extra arguments for the distance computing method

    Attributes
    ----------
    ts_attr: str
        The name of the time series attribute used by this quality measure (get only property)

    References
    ----------
    [1] Page 215-216 (3 Quality measures)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """
    def __init__(self, df: DataFrame = DataFrame(), **dist_kwargs) -> None:
        QualityMeasure.__init__(self, df)
        TSDistance.__init__(self, **dist_kwargs)
        self.df = df # trigger internal update

    @property
    def ts_attr(self):
        return self.model_attributes[0]

    @QualityMeasure.df.setter
    def df(self, df: DataFrame):
        QualityMeasure.df.fset(self, df)
        if len(df.columns) > 0:
            self.dataset_model = self.compute_model(df[self.ts_attr])

    def model_from_subgroup(self, sg: Subgroup):
        """
        Compute an average model for the time series extracted from the specified subroup

        Parameters
        ----------
        sg: Subgroup
            The subgroup for which to compute the model

        Returns
        -------
        numpy.array of shape (sz, d):
            The computed model for the subgroup
        """
        return self.compute_model(self.df.loc[sg.cover][self.ts_attr])

    def model_from_cover(self, sg: pandas.Index):
        """
        Compute an average model for the time series extracted based on the specified index

        Parameters
        ----------
        sg: pandas.Index
            The index to use to obtain a subset of the time series and compute the model

        Returns
        -------
        numpy.array of shape (sz, d):
            The computed model for the corresponding subgroup
        """
        return self.compute_model(self.df.loc[sg][self.ts_attr])


    def compute_quality(self, sg: DataFrame):
        quality = 0
        if not sg.empty:
            sg_model = self.compute_model(sg[self.ts_attr])
            quality = pow(len(sg),0.5) * self.measure_distance(sg_model.ravel(), self.dataset_model.ravel())
        return quality


class EubModel(TSModel):
    """A wrapper class around euclidean barycenter averaging method, implementation from tslearn.barycenters.euclidean_barycenter"""
    def compute_model(self, series: Series):
        return eub(series.to_numpy())

class DBAModel(TSModel):
    """A wrapper class around dynamic DTW barycenter averaging method, implementation from tslearn.barycenters.dtw_barycenter_averaging"""
    def compute_model(self, series: Series):
        return dba(series.to_numpy())

class EuclideanDistance(TSDistance):
    """A wrapper class around euclidean distance method"""
    def measure_distance(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(c1, c2))) # taken from python official doc to use instead of required to have python 3.8 as this is when the dist function was introduced

class DtwDistance(TSDistance):
    """A wrapper class around the DTW distance method, implementation from tslearn.metrics.dtw"""
    def measure_distance(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return dtw(c1, c2, **self.dist_kwargs)

class FastDtwDistance(TSDistance):
    """A wrapper class around the DTW distance method, implementation from fastdtw.fastdtw"""
    def measure_distance(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return fastdtw(c1, c2, **self.dist_kwargs)[0] # maybe this is slow because it is also computing the path ?


class DtwDba(TSQuality, DtwDistance, DBAModel): 
    """
    A time series quality measure based on : \n
    - distance = dtw(from tslearn.metrics.dtw)\n
    - model = DTW barycenter averaging or DBA (from tslearn.barycenters.dtw_barycenter_averaging)\n
    Notes: This method is potentially very slow because of the complexity of DBA

    See also
    --------
    tslearn.metrics.dtw,tslearn.barycenters.dtw_barycenter_averaging

    Examples
    --------
    >>> from skmine.dssd import DtwDba
    >>> import pandas
    >>> import numpy as np
    >>> s1 = np.array([1, 2, 6, 5, 7])
    >>> df = pandas.DataFrame({"ts": [s1, s1, s1]})
    >>> DtwDba(df[["ts"]]).compute_quality(df)
    0.0
    """
    
    def __str__(self) -> str:
        return "dtw-dba"

class FastDtwDba(TSQuality, FastDtwDistance, DBAModel):
    """
    A time series quality measure based on : \n
    - distance = dtw(from fastdtw.fastdtw) \n
    - model = DTW barycenter averaging or DBA (from tslearn.barycenters.dtw_barycenter_averaging)\n
    Notes: This method is potentially very slow because of the complexity of DBA

    See also
    --------
    fastdtw.fastdtw, tslearn.barycenters.dtw_barycenter_averaging

    Examples
    --------
    >>> from skmine.dssd import FastDtwDba
    >>> from scipy.spatial.distance import euclidean
    >>> import pandas
    >>> import numpy as np
    >>> s1 = np.array([1, 2, 6, 5, 7])
    >>> df = pandas.DataFrame({"ts": [s1, s1, s1]})
    >>> FastDtwDba(df[["ts"]], dist=euclidean).compute_quality(df) # passing distance args dist=euclidean to control how fastdtw works
    0.0
    """

    def __str__(self) -> str:
        return "fastdtw-dba"

class EuclideanEub(TSQuality, EuclideanDistance, EubModel):
    """
    A time series quality measure based on :\n
    - distance = standard euclidean distance(from `fastdtw.fastdtw`)\n
    - model = euclidean barycenter (from `tslearn.barycenters.euclidean_barycenter`)

    See also
    --------
    tslearn.barycenters.euclidean_barycenter

    Examples
    --------
    >>> from skmine.dssd import EuclideanEub
    >>> import pandas
    >>> import numpy as np
    >>> s1 = np.array([1, 2, 6, 5, 7])
    >>> df = pandas.DataFrame({"ts": [s1, s1, s1]})
    >>> EuclideanEub(df[["ts"]]).compute_quality(df)
    0.0
    """

    def __str__(self) -> str:
        return "euclidean-eub"
