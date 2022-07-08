from collections import defaultdict
import math
from tslearn.metrics import dtw
from tslearn.barycenters import dtw_barycenter_averaging as dba
from tslearn.barycenters import euclidean_barycenter as eub
import numpy as np
import pandas
import pytest
from .. import quality_measures as qa
from ..utils import column_shares


def test_quality_measures_factory():
    qa.register("undefined_quality_measure", lambda x,y={}: "")
    assert qa.create("undefined_quality_measure", pandas.DataFrame(), {}) == ""
    qa.register("undefined_quality_measure", None)

    with pytest.raises(ValueError):
        qa.create("undefined_quality_measure", pandas.DataFrame(), {})


def test_wracc():
    df = pandas.DataFrame({"a": [True, True, True, False] * 2})

    with pytest.raises(ValueError):
        qa.WRACCQuality.ones_fraction(pandas.DataFrame(), "example_attribute")

    assert qa.WRACCQuality.ones_fraction(df, "a") == (3 * 2) / 8

    sg = pandas.DataFrame({ "a": [True, True, True, True] })
    q = qa.create("wracc", entire_df=df, extra_parameters={"binary_model_attribute": "a"})

    assert q.compute_quality(df) == 0

    assert qa.WRACCQuality.ones_fraction(sg, "a") == 4 / 4
    assert q.compute_quality(sg) == (4 / 8) * abs(1 - 3 * 2 / 8)



def test_kl_quality():
    subset_correct_distribution = defaultdict(int, {"one": .75, "two": .25})
    entire_dataset_distribution = defaultdict(int, {"one": .5, "two": .5})

    assert qa.KLQuality.kl(defaultdict(int), entire_dataset_distribution) == 0
    res = .75 * math.log2(.75/.5) + .25 * math.log2(.25/.5)
    assert qa.KLQuality.kl(subset_correct_distribution, entire_dataset_distribution) == res


    df = pandas.DataFrame({
        "a": ["one","one","two","two"] * 2,
        "bin": [True, True, False, False] * 2
    })
    
    sg = pandas.DataFrame({
        "a": ["one","one","one","two"],
        "bin": [True, True, False, False]
    })
    

    assert qa.KLQuality(df, []).compute_quality(sg) == 0

    kl_a = .75 * math.log2(.75/.5) + .25 * math.log2(.25/.5)
    assert qa.KLQuality(df, ["a"]).compute_quality(sg) == kl_a

    kl_b = .5 * math.log2(.5/.5) + .5 * math.log2(.5/.5)
    assert qa.KLQuality(df, ["a", "bin"]).compute_quality(sg) == kl_a + kl_b


    kl: qa.KLQuality = qa.create("kl", entire_df=df, extra_parameters={"model_attributes": ["a", "bin"] })
    assert kl.compute_quality(sg) == kl_a + kl_b

    wkl: qa.KLQuality = qa.create("wkl", entire_df=df, extra_parameters={"model_attributes": ["a", "bin"] })
    assert wkl.compute_quality(sg) == (kl_a + kl_b) * len(sg)


def test_measure_distance():
    s1 = np.array([1, 2, 6, 5, 7])
    s2 = np.array([1, 2, 5, 5, 7])
    s3 = np.array([0, 1, 2, 2, 4, 5, 7, 8])

    # testing same length time series for euclidean
    assert qa.EuclideanDistance().measure_distance(s1, s1) == 0

    assert qa.EuclideanDistance().measure_distance(s1, s2) == 1

    # testing same and different length time series for dtw 
    assert qa.FastDtwDistance().measure_distance(s1, s1) == 0

    # vvv--- for the tslearns.metrics.dtw version
    # assert qa.measure_distance(s1, s3, "dtw") == 2.449489742783178
    assert qa.FastDtwDistance().measure_distance(s1, s3) == 4.0


def test_time_series_model():
    s1 = np.array([1, 2, 6, 5, 7])
    s2 = np.array([1, 2, 5, 5, 7])
    s3 = np.array([0, 1, 2, 2, 4])
    df = pandas.DataFrame({"a": [s1, s2, s3]})

    # ensuring that the correct function is being used depending on the specified target_model
    assert np.array_equal(qa.EubModel().compute_model(df, "a"), eub(df["a"].to_numpy()))

    assert np.array_equal(qa.DBAModel().compute_model(df, "a"), dba(df["a"].to_numpy()))


def test_ts_quality():
    s1 = np.array([1, 2, 6, 5, 7])
    s2 = np.array([1, 2, 5, 5, 7])
    s3 = np.array([0, 1, 2, 2, 4])
    df = pandas.DataFrame({"ts": [s1, s2, s3]})
    sg = pandas.DataFrame({"ts": [s3]})

    # s: qa.TSQuality = qa.create("ts_quality", entire_df=df, extra_parameters={"model_attribute": "ts", "target_model": "eub", "dist_measure": "euclidean"})
    # s: qa.TSQuality = qa.create("ts_quality", entire_df=df, extra_parameters={"model_attribute": "ts"})
    s = qa.EuclideanEubTSQuality(df, "ts")
    # ensure an empty subgroup has a zero quality
    assert s.compute_quality(pandas.DataFrame()) == 0

    # ensure a subgroup no different from the entire dataset has zero quality
    assert s.compute_quality(df) == 0

    # ensure the specified formula is actually being used for computing quality
    model = qa.EubModel().compute_model(sg, "ts")
    assert s.compute_quality(sg) == pow(len(sg),0.5) * qa.EuclideanDistance().measure_distance(s.dataset_model, model)
