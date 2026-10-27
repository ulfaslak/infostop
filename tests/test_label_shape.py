import numpy as np
import pytest

from infostop import Infostop
from infostop.models import SpatialInfomap


def make_trace(seed):
    """Home (rows 0-29), work (30-59), home (60-89): Gaussian blobs of points,
    one point per 60 s."""
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


def simulate_numpy_2_0_0_unique(monkeypatch):
    """Make `np.unique(..., return_inverse=True, axis=...)` return the inverse with
    shape (N, 1), as NumPy 2.0.0 does. Returns a list that counts the calls."""
    real_unique = np.unique
    hits = []

    def unique(ar, *args, **kwargs):
        out = real_unique(ar, *args, **kwargs)
        if kwargs.get("return_inverse") and kwargs.get("axis") is not None:
            assert not args and not kwargs.get("return_index")
            out = (out[0], out[1].reshape(-1, 1)) + tuple(out[2:])
            hits.append(1)
        return out

    monkeypatch.setattr(np, "unique", unique)
    return hits


FITS = {
    "single": lambda kw: (Infostop(**kw), make_trace(0)),
    "single 2 columns": lambda kw: (Infostop(**kw), make_trace(0)[:, :2]),
    "multi": lambda kw: (Infostop(**kw), [make_trace(0), make_trace(1)]),
    "spatial": lambda kw: (SpatialInfomap(**kw), make_trace(0)[:, :2]),
}


def fit(name, min_spacial_resolution):
    model, data = FITS[name](dict(min_spacial_resolution=min_spacial_resolution))
    labels = model.fit_predict(data)
    return (labels, data) if isinstance(data, list) else ([labels], [data])


def check(labels, data):
    for labels_u, data_u in zip(labels, data):
        assert labels_u.shape == (len(data_u),)
        # Labels line up with the rows: both home visits share a stop, work is another
        home = set(labels_u[:30].tolist()) | set(labels_u[60:].tolist())
        work = set(labels_u[30:60].tolist())
        assert len(home - {-1}) == 1 and len(work - {-1}) == 1
        assert not (home & work) - {-1}


@pytest.mark.parametrize("min_spacial_resolution", [0, 1e-4])
@pytest.mark.parametrize("name", FITS)
def test_labels_are_1d_and_aligned(name, min_spacial_resolution):
    check(*fit(name, min_spacial_resolution))


@pytest.mark.parametrize("min_spacial_resolution", [0, 1e-4])
@pytest.mark.parametrize("name", FITS)
def test_numpy_2_0_0_unique_gives_same_labels(
    name, min_spacial_resolution, monkeypatch
):
    expected, data = fit(name, min_spacial_resolution)
    hits = simulate_numpy_2_0_0_unique(monkeypatch)
    labels, _ = fit(name, min_spacial_resolution)
    assert hits  # the simulation was used
    check(labels, data)
    for a, b in zip(labels, expected):
        np.testing.assert_array_equal(a, b)
