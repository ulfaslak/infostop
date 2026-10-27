import numpy as np
import pytest

from infostop import Infostop
from infostop.models import SpatialInfomap


def make_trace(seed):
    """Home, work, home: Gaussian blobs of points, one point per 60 s."""
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


@pytest.fixture
def numpy_2_0_0_unique(monkeypatch):
    """Make `np.unique(..., return_inverse=True, axis=...)` return the inverse with
    shape (N, 1), as NumPy 2.0.0 does."""
    real_unique = np.unique

    def unique(ar, *args, **kwargs):
        out = real_unique(ar, *args, **kwargs)
        if kwargs.get("return_inverse") and kwargs.get("axis") is not None:
            assert not kwargs.get("return_index")
            out = (out[0], out[1].reshape(-1, 1)) + tuple(out[2:])
        return out

    monkeypatch.setattr(np, "unique", unique)


FITS = {
    "single": lambda kw: (Infostop(**kw), make_trace(0)),
    "single 2 columns": lambda kw: (Infostop(**kw), make_trace(0)[:, :2]),
    "multi": lambda kw: (Infostop(**kw), [make_trace(0), make_trace(1)]),
    "spatial": lambda kw: (SpatialInfomap(**kw), make_trace(0)[:, :2]),
}


def check_shapes(fit, min_spacial_resolution):
    model, data = FITS[fit](dict(min_spacial_resolution=min_spacial_resolution))
    labels = model.fit_predict(data)
    for labels_u, data_u in zip(
        labels if isinstance(data, list) else [labels],
        data if isinstance(data, list) else [data],
    ):
        assert labels_u.shape == (len(data_u),)
        assert len(set(labels_u.tolist()) - {-1}) > 0


@pytest.mark.parametrize("min_spacial_resolution", [0, 1e-4])
@pytest.mark.parametrize("fit", FITS)
def test_labels_are_1d(fit, min_spacial_resolution):
    check_shapes(fit, min_spacial_resolution)


@pytest.mark.parametrize("min_spacial_resolution", [0, 1e-4])
@pytest.mark.parametrize("fit", FITS)
def test_labels_are_1d_with_numpy_2_0_0_unique(
    fit, min_spacial_resolution, numpy_2_0_0_unique
):
    check_shapes(fit, min_spacial_resolution)
