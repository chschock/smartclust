import pytest
import numpy as np
from scipy.cluster.hierarchy import linkage
from smartcluster import flatten


def test_default():
    vectors = np.random.random((1000, 10)).round(1)

    Z = linkage(vectors, metric="euclidean", method="average")
    clusters, score = flatten(Z)

    assert len(clusters) == len(vectors)
    assert score > 0
