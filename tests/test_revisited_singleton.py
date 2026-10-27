import numpy as np
import pytest

from infostop import Infostop
from infostop.models import SpatialInfomap

HOME, WORK, GYM = (55.70, 12.50), (55.71, 12.52), (55.69, 12.55)


def make_trace(seed, places, n_per_visit, noise=2e-5):
    """Gaussian blobs of points at each of `places`, joined by moving segments
    (10 points along a straight line), one point per 60 s. Returns the trace and
    the rows of each visit."""
    rng = np.random.default_rng(seed)
    parts, rows, t, i0 = [], [], 0, 0
    for i, ((lat, lon), n) in enumerate(zip(places, n_per_visit)):
        if i > 0:
            a, b = np.array(places[i - 1]), np.array([lat, lon])
            f = np.linspace(0, 1, 12)[1:-1, None]
            parts.append(np.column_stack([a + f * (b - a), t + 60 * np.arange(10)]))
            t, i0 = t + 600, i0 + 10
        parts.append(
            np.column_stack(
                [
                    lat + rng.normal(0, noise, n),
                    lon + rng.normal(0, noise, n),
                    t + 60 * np.arange(n),
                ]
            )
        )
        rows.append(slice(i0, i0 + n))
        t, i0 = t + 60 * n + 60, i0 + n
    return np.vstack(parts), rows


def labels_at(labels, rows):
    return set(labels[rows].tolist())


@pytest.mark.parametrize("min_spacial_resolution", [1e-4, 1e-3])
def test_visits_merged_by_rounding_get_a_label(min_spacial_resolution):
    # Home, work, home: both home visits round to one coordinate
    trace, rows = make_trace(0, [HOME, WORK, HOME], [30, 30, 30])
    model = Infostop(
        label_singleton=False, min_spacial_resolution=min_spacial_resolution
    )
    labels = model.fit_predict(trace)

    home = labels_at(labels, rows[0]) | labels_at(labels, rows[2])
    assert len(home) == 1 and -1 not in home
    assert labels_at(labels, rows[1]) == {-1}  # work is visited once
    assert sorted(model._counts.tolist()) == [1, 2]


def test_visits_with_identical_medians_get_a_label():
    # A device that reports the same rounded coordinates throughout a stay gives
    # identical medians for every visit, without any `min_spacial_resolution`.
    # The gym is visited twice with medians a few metres apart: an Infomap module.
    trace, rows = make_trace(
        1, [HOME, GYM, HOME, WORK, GYM], [30, 20, 25, 30, 20], noise=0
    )
    gym_noise = np.random.default_rng(2).normal(0, 2e-5, (40, 2))
    gym_rows = np.r_[rows[1], rows[4]]
    trace[gym_rows, :2] += gym_noise

    model = Infostop(label_singleton=False)
    labels = model.fit_predict(trace)

    home = labels_at(labels, rows[0]) | labels_at(labels, rows[2])
    gym = labels_at(labels, rows[1]) | labels_at(labels, rows[4])
    assert len(home) == 1 and -1 not in home
    assert len(gym) == 1 and -1 not in gym
    assert home != gym
    assert labels_at(labels, rows[3]) == {-1}  # work is visited once


def test_spatial_infomap_points_merged_by_rounding_get_a_label():
    rng = np.random.default_rng(3)
    home = HOME + rng.normal(0, 2e-6, (30, 2))  # all round to one coordinate
    gym = GYM + rng.normal(0, 2e-5, (30, 2))  # a few distinct coordinates
    work = np.array([WORK])  # a single point
    model = SpatialInfomap(label_singleton=False, min_spacial_resolution=1e-4)
    labels = model.fit_predict(np.vstack([home, gym, work]))

    assert len(set(labels[:30])) == 1 and labels[0] != -1
    assert -1 not in labels[30:60] and labels[0] not in labels[30:60]
    assert labels[60] == -1


@pytest.mark.parametrize("min_spacial_resolution", [0, 1e-4])
def test_single_visits_stay_unlabelled(min_spacial_resolution):
    trace, rows = make_trace(4, [HOME, WORK, GYM], [30, 30, 30])
    labels = Infostop(
        label_singleton=False, min_spacial_resolution=min_spacial_resolution
    ).fit_predict(trace)
    assert set(labels.tolist()) == {-1}
