"""Characterization tests: pin the partitions the current pipeline produces.

Label integers are arbitrary (they follow Infomap's module numbering, which starts at
1, except when Infomap is not run at all), so partitions are compared after
relabelling by order of first occurrence. Positions labelled `-1` must match exactly.

Each pinned partition is written as runs of `(label, count)` over the trace, so it
reads like the trace design: a stop location is one run, a walk between stops is
a run of `-1`.

Every trace is designed so that its partition does not depend on Infomap's random
number stream. On borderline data (stops that overlap within `r2`), Infomap 2.x
gives different labels on macOS and on Linux, and different Infomap versions give
different labels, so borderline cases are deliberately not pinned here.
"""

import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap

# Geo traces are laid out in metres around this point and converted to lat/lon.
LAT0, LON0 = 55.6761, 12.5683
M_PER_DEG_LAT = 6371000 * np.pi / 180

# Stop locations, (east, north) in metres. Far apart compared to the default
# `r1 = r2 = 10`, so the partition does not depend on the noise realisation.
A = (0, 0)
B = (400, 0)
C = (0, 300)
D = (400, 300)

# One user. A is visited three times (plus a 3-minute visit, shorter than the
# default `min_staying_time`), B twice, C and D once. D ends the trace.
SINGLE_VISITS = [(A, 20), (B, 20), (A, 20), (C, 20), (A, 4), (B, 20), (A, 20), (D, 20)]

# Three users sharing stop locations. User 1 only walks and never stops.
MULTI_VISITS = [
    [(A, 20), (B, 20), (A, 20)],
    [(A, 1), (D, 1)],
    [(B, 15), (C, 20), (B, 15), (C, 20), (D, 20)],
]


def make_trace(seed, visits, metric="haversine", with_time=True, noise=2.0, step=50.0):
    """Trace that stays at each visit in turn and walks in a straight line between.

    `visits` is a list of `((x, y), n)`: `n` points scattered around `(x, y)` metres
    with Gaussian noise of std `noise`. Walks have one point per `step` metres.
    Points are 60 seconds apart. Returns lat/lon for `metric="haversine"` and the
    metres themselves for `metric="euclidean"`, plus a time column if `with_time`.
    """
    rng = np.random.default_rng(seed)
    xy = []
    for k, ((x, y), n) in enumerate(visits):
        if k > 0:
            (x0, y0), _ = visits[k - 1]
            n_steps = int(np.hypot(x - x0, y - y0) // step)
            for f in np.arange(1, n_steps) / n_steps:
                xy.append((x0 + f * (x - x0), y0 + f * (y - y0)))
        xy.extend(zip(x + rng.normal(0, noise, n), y + rng.normal(0, noise, n)))

    coords = to_coords(np.array(xy), metric)
    if with_time:
        coords = np.column_stack([coords, 60.0 * np.arange(len(coords))])
    return coords


def blob_points(seed, metric="haversine", noise=2.0):
    """Point cloud for `SpatialInfomap`: 30-point blobs at A, B and D, one point at C."""
    rng = np.random.default_rng(seed)
    blobs = [(A, 30), (B, 30), (C, 1), (D, 30)]
    xy = np.vstack([np.array(c) + rng.normal(0, noise, (n, 2)) for c, n in blobs])
    return to_coords(xy, metric)


def to_coords(xy, metric):
    """Metres (east, north) to lat/lon for haversine; unchanged for euclidean."""
    if metric == "euclidean":
        return xy
    lat = LAT0 + xy[:, 1] / M_PER_DEG_LAT
    lon = LON0 + xy[:, 0] / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))
    return np.column_stack([lat, lon])


def canonical(labels):
    """Relabel by order of first occurrence, keeping `-1` fixed."""
    mapping = {-1: -1}
    out = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping) - 1
        out.append(mapping[label])
    return out


def runs(labels):
    """Run-length encode a label sequence as `[(label, count), ...]`."""
    out = []
    for label in labels:
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1] + 1)
        else:
            out.append((label, 1))
    return out


def partition(labels):
    """Canonical runs of one label array, or of a list of them in a shared label space."""
    if isinstance(labels, list):
        flat = canonical(np.concatenate(labels).tolist())
        bounds = np.cumsum([0] + [len(l) for l in labels])
        return [runs(flat[i:j]) for i, j in zip(bounds[:-1], bounds[1:])]
    return runs(canonical(labels.tolist()))


def test_canonical_keeps_outliers_and_orders_by_first_occurrence():
    assert canonical([7, -1, 3, 7, -1, 0, 3]) == [0, -1, 1, 0, -1, 2, 1]
    assert runs([0, 0, -1, 1, 1, 1, 0]) == [(0, 2), (-1, 1), (1, 3), (0, 1)]


# Pinned partitions are laid out to follow the trace, so black leaves them alone.
# fmt: off

# A: 3 visits, B: 2, C and D: 1 each (singletons, so they get their own labels).
SINGLE_WITH_TIME = [
    (0, 20), (-1, 7), (1, 20), (-1, 7), (0, 20), (-1, 5), (2, 20),
    (-1, 5 + 4 + 7),  # walk C->A, the 3-minute visit to A, walk A->B
    (1, 20), (-1, 7), (0, 20), (-1, 9), (3, 20),
]
# Without time the 3-minute visit to A is a stop event like the others.
SINGLE_NO_TIME = [
    (0, 20), (-1, 7), (1, 20), (-1, 7), (0, 20), (-1, 5), (2, 20),
    (-1, 5), (0, 4), (-1, 7),
    (1, 20), (-1, 7), (0, 20), (-1, 9), (3, 20),
]

# fmt: on


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
@pytest.mark.parametrize(
    "with_time, expected", [(True, SINGLE_WITH_TIME), (False, SINGLE_NO_TIME)]
)
def test_infostop_single_user(metric, with_time, expected):
    trace = make_trace(1, SINGLE_VISITS, metric, with_time)
    labels = Infostop(distance_metric=metric).fit_predict(trace)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(trace),)
    assert partition(labels) == expected


@pytest.mark.parametrize(
    "first_longitude, middle_longitude",
    [(179.99999, -179.99999), (-179.99999, 179.99999)],
    ids=["east-first", "west-first"],
)
def test_infostop_repeated_antimeridian_visits_share_one_label(
    first_longitude, middle_longitude
):
    longitudes = (
        [first_longitude] * 20 + [middle_longitude] * 20 + [first_longitude] * 20
    )
    trace = np.column_stack([np.full(60, 10.0), longitudes, 60.0 * np.arange(60)])

    labels = Infostop().fit_predict(trace)

    assert labels[0] >= 0
    assert np.all(labels == labels[0])


def test_western_stop_keeps_one_label_with_min_spacial_resolution():
    # Two visits to one place in New York. The first event's median lies exactly
    # halfway between two 1e-5 cells; a median off in its last bit is rounded into
    # the other cell, away from the second visit, and the place gets two labels.
    trace = np.array(
        [
            [40.7, -73.98121, 0],
            [40.7, -73.98122, 60],
            [40.8, -73.0, 120],
            [40.7, -73.98121, 180],
            [40.7, -73.98121, 240],
        ]
    )
    model = Infostop(r1=10, r2=0.5, min_staying_time=1, min_spacial_resolution=1e-5)

    labels = model.fit_predict(trace)

    assert partition(labels) == [(0, 2), (-1, 1), (0, 2)]


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
@pytest.mark.parametrize("with_time", [True, False])
def test_infostop_multi_user(metric, with_time):
    traces = [
        make_trace(10 + u, visits, metric, with_time)
        for u, visits in enumerate(MULTI_VISITS)
    ]
    labels = Infostop(distance_metric=metric).fit_predict(traces)

    assert isinstance(labels, list)
    assert [len(l) for l in labels] == [len(t) for t in traces]
    # Labels are shared across users: B is label 1 for both user 0 and user 2.
    # fmt: off
    assert partition(labels) == [
        [(0, 20), (-1, 7), (1, 20), (-1, 7), (0, 20)],
        [(-1, 11)],
        [(1, 15), (-1, 9), (2, 20), (-1, 9), (1, 15), (-1, 9), (2, 20), (-1, 7), (3, 20)],
    ]
    # fmt: on


def test_infostop_label_singleton_false():
    trace = make_trace(1, SINGLE_VISITS)
    labels = Infostop(label_singleton=False).fit_predict(trace)

    # C and D are visited once, so they become -1 and merge with the walks.
    # fmt: off
    assert partition(labels) == [
        (0, 20), (-1, 7), (1, 20), (-1, 7), (0, 20),
        (-1, 5 + 20 + 5 + 4 + 7),
        (1, 20), (-1, 7), (0, 20),
        (-1, 9 + 20),
    ]
    # fmt: on


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
@pytest.mark.parametrize("weight_exponent", [1, 2])
def test_infostop_weighted_keeps_separate_stops(metric, weight_exponent):
    # Stops with no links between them stay separate under any weighting.
    trace = make_trace(1, SINGLE_VISITS, metric)
    model = Infostop(
        distance_metric=metric, weighted=True, weight_exponent=weight_exponent
    )

    assert partition(model.fit_predict(trace)) == SINGLE_WITH_TIME


# Two triangles of spots with 1 m sides, 4 m apart, each spot visited once and
# followed by a visit to D. The trace has no noise, so every stop event sits exactly
# on its spot. All six spots are within `r2` of each other. Weighting links by
# `1 / distance ** weight_exponent` makes links within a triangle stronger than
# links across; the split happens between exponent 1 and 2 for gaps of about 3 to
# 5.5 m, so 4 m keeps a margin on both sides.
TRIANGLE = [(0, 0), (1, 0), (0.5, 0.87)]
TRIANGLE_SPOTS = TRIANGLE + [(x + 4, y) for x, y in TRIANGLE]
TRIANGLE_VISITS = [v for spot in TRIANGLE_SPOTS for v in ((spot, 10), (D, 10))]


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
@pytest.mark.parametrize(
    "weighted, weight_exponent, split",
    [(False, 1, False), (True, 1, False), (True, 2, True)],
)
def test_infostop_weighting_splits_close_spots(
    metric, weighted, weight_exponent, split
):
    trace = make_trace(0, TRIANGLE_VISITS, metric, noise=0)
    model = Infostop(
        distance_metric=metric, weighted=weighted, weight_exponent=weight_exponent
    )
    labels = model.fit_predict(trace)

    # One line per spot: the visit, the walk to D, D, the walk to the next spot.
    left, d, right = 0, 1, (2 if split else 0)
    # fmt: off
    assert partition(labels) == [
        (left, 10), (-1, 9), (d, 10), (-1, 8),
        (left, 10), (-1, 8), (d, 10), (-1, 8),
        (left, 10), (-1, 8), (d, 10), (-1, 8),
        (right, 10), (-1, 8), (d, 10), (-1, 8),
        (right, 10), (-1, 8), (d, 10), (-1, 8),
        (right, 10), (-1, 8), (d, 10),
    ]
    # fmt: on


# Three spots 15 m apart (more than `r2`), each visited twice with D in between.
# All three lie inside one 0.001-degree grid cell, at least 10 m from its edges.
GRID_SPOTS = [(-40, -10), (-25, -10), (-10, -10)]
GRID_VISITS = [v for spot in GRID_SPOTS * 2 for v in ((spot, 10), (D, 10))]


@pytest.mark.parametrize("resolution", [0, 1e-3])
def test_infostop_min_spacial_resolution(resolution):
    trace = make_trace(6, GRID_VISITS)
    labels = Infostop(min_spacial_resolution=resolution).fit_predict(trace)

    # Rounding to the grid puts the three spots on one point, so they share a label.
    s1, d, s2, s3 = (0, 1, 2, 3) if resolution == 0 else (0, 1, 0, 0)

    def visit(s):
        return [(s, 10), (-1, 9), (d, 10), (-1, 9)]

    assert partition(labels) == ((visit(s1) + visit(s2) + visit(s3)) * 2)[:-1]


@pytest.mark.parametrize("metric", ["haversine", "euclidean"])
@pytest.mark.parametrize(
    "label_singleton, expected",
    [
        (True, [(0, 30), (1, 30), (2, 1), (3, 30)]),
        (False, [(0, 30), (1, 30), (-1, 1), (2, 30)]),
    ],
)
def test_spatial_infomap(metric, label_singleton, expected):
    points = blob_points(3, metric)
    model = SpatialInfomap(distance_metric=metric, label_singleton=label_singleton)
    labels = model.fit_predict(points)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(points),)
    assert partition(labels) == expected
