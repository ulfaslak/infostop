import numpy as np
import pytest

from infostop import Infostop
from infostop.models import SpatialInfomap


def make_trace(seed=0):
    """Home, work, home: two Gaussian blobs of points, 60 s apart within a visit."""
    rng = np.random.default_rng(seed)

    def visit(lat, lon, t0, n=30):
        return np.column_stack(
            [
                lat + rng.normal(0, 2e-5, n),
                lon + rng.normal(0, 2e-5, n),
                t0 + 60 * np.arange(n),
            ]
        )

    return np.vstack(
        [visit(55.70, 12.50, 0), visit(55.71, 12.52, 4000), visit(55.70, 12.50, 8000)]
    )


def nan_message(n):
    return rf"^There are {n} NaN values in the input data\.$"


def test_infostop_nan_message_gives_count():
    trace = make_trace()
    trace[5, 0] = np.nan
    with pytest.raises(AssertionError, match=nan_message(1)):
        Infostop().fit_predict(trace)


def test_infostop_nan_message_counts_all_columns():
    trace = make_trace()
    trace[5, 0] = np.nan
    trace[6, 1] = np.nan
    trace[7, 2] = np.nan
    with pytest.raises(AssertionError, match=nan_message(3)):
        Infostop().fit_predict(trace)


def test_infostop_nan_message_counts_across_users():
    trace_a, trace_b = make_trace(0), make_trace(1)
    trace_a[5, 0] = np.nan
    trace_b[10:12, 1] = np.nan
    with pytest.raises(AssertionError, match=nan_message(3)):
        Infostop().fit_predict([trace_a, trace_b])


def test_spatial_infomap_nan_message_gives_count():
    coords = make_trace()[:, :2]
    coords[5, 0] = np.nan
    coords[6, 1] = np.nan
    with pytest.raises(AssertionError, match=nan_message(2)):
        SpatialInfomap().fit_predict(coords)
