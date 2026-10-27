"""`num_trials`: Infomap keeps the best of several runs, so labels on crowded stops
do not depend on the Infomap version or the operating system.

The crowded traces below put 3 to 6 spots inside a 30 x 30 m square (default
`r2 = 10`), so stops overlap and a single Infomap run often ends in a worse
partition. Which worse partition depends on Infomap's random numbers, and those
differ between Infomap versions and between macOS and Linux. The pinned partitions
are the 50-trial results, which were identical on Infomap 1.0.6, 2.3.0, 2.13.0 and
2.16.0, on macOS and Linux. With 1 trial, every one of those setups gave a
different partition, so the tests fail if the default stops passing `num_trials`.
"""

import inspect

import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap

LAT0, LON0 = 55.6761, 12.5683
M_PER_DEG_LAT = 6371000 * np.pi / 180
FAR = (300, 300)


def to_latlon(xy):
    lat = LAT0 + xy[:, 1] / M_PER_DEG_LAT
    lon = LON0 + xy[:, 0] / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))
    return np.column_stack([lat, lon])


def make_trace(seed, visits, step=50.0):
    """Visit each `((x, y), n)` in turn (n points, 2 m noise), walking in between."""
    rng = np.random.default_rng(seed)
    xy = []
    for k, ((x, y), n) in enumerate(visits):
        if k > 0:
            (x0, y0), _ = visits[k - 1]
            n_steps = int(np.hypot(x - x0, y - y0) // step)
            for f in np.arange(1, n_steps) / n_steps:
                xy.append((x0 + f * (x - x0), y0 + f * (y - y0)))
        xy.extend(zip(x + rng.normal(0, 2, n), y + rng.normal(0, 2, n)))
    coords = to_latlon(np.array(xy))
    return np.column_stack([coords, 60.0 * np.arange(len(coords))])


def crowded(seed):
    """A crowded trace (lat, lon, time) and a crowded point cloud (metres)."""
    rng = np.random.default_rng(1000 + seed)
    spots = rng.uniform(0, 30, (rng.integers(3, 7), 2))
    visits = []
    for _ in range(25):
        c = spots[rng.integers(len(spots))] + rng.normal(0, 2, 2)
        visits += [((c[0], c[1]), int(rng.integers(5, 16))), (FAR, 6)]
    trace = make_trace(2000 + seed, visits)
    centres = rng.uniform(0, 40, (rng.integers(4, 9), 2))
    points = np.vstack(
        [c + rng.normal(0, 3, (int(rng.integers(10, 31)), 2)) for c in centres]
    )
    return trace, points


def stop_labels(labels):
    """Label of each stop event in order, relabelled by first occurrence.

    Which points are -1 is decided before Infomap runs, so the labels of the stop
    events are what `num_trials` can change.
    """
    mapping, out = {}, []
    previous = None
    for label in labels.tolist():
        if label != previous and label != -1:
            out.append(mapping.setdefault(label, len(mapping)))
        previous = label
    return out


def point_labels(labels):
    """Labels relabelled by first occurrence."""
    mapping = {}
    return [mapping.setdefault(label, len(mapping)) for label in labels.tolist()]


def runs(values):
    out = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((v, 1))
    return out


# Stop-event labels of three crowded traces with `weighted=True`. (On the unweighted
# versions of these traces, a single trial already found the best partition on at
# least one setup, so they would not show the default at work.)
# fmt: off
INFOSTOP_BEST_OF_50 = {
    77: [0, 1, 2, 1, 3, 1, 2, 1, 0, 1, 2, 1, 0, 1, 3, 1, 4, 1, 2, 1, 1, 3, 1, 2, 1,
         2, 1, 4, 1, 0, 1, 0, 1, 1, 4, 1, 4, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 1],
    25: [0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 3, 0, 3, 0, 1, 0, 3, 0, 3, 0, 3, 0, 4, 0, 2,
         0, 3, 0, 4, 0, 2, 0, 2, 0, 0, 4, 0, 0, 5, 0, 5, 0, 2, 0, 5, 0],
    73: [0, 1, 2, 1, 1, 3, 1, 3, 1, 0, 1, 4, 1, 1, 0, 1, 0, 1, 2, 1, 3, 1, 0, 1, 0,
         1, 3, 1, 1, 5, 1, 0, 1, 4, 1, 4, 1, 2, 1, 4, 1, 5, 1, 1, 1],
}
SPATIAL_BEST_OF_50 = {
    29: [(0, 29), (1, 1), (2, 2), (1, 1), (2, 5), (1, 1), (2, 4), (3, 14), (1, 20), (3, 14)],
    47: [(0, 12), (1, 28), (2, 17), (3, 18), (1, 1), (3, 23), (2, 1), (1, 16)],
    74: [(0, 18), (1, 14), (2, 1), (3, 1), (2, 1), (3, 1), (2, 1), (3, 2), (2, 1), (3, 1),
         (2, 1), (3, 12), (4, 29), (3, 18), (1, 11), (2, 23), (4, 2), (2, 4)],
}
# fmt: on


@pytest.mark.parametrize("seed", sorted(INFOSTOP_BEST_OF_50))
def test_infostop_default_keeps_the_best_of_50_trials(seed):
    trace, _ = crowded(seed)
    expected = INFOSTOP_BEST_OF_50[seed]
    assert stop_labels(Infostop(weighted=True).fit_predict(trace)) == expected
    assert stop_labels(Infostop(weighted=True, num_trials=50).fit_predict(trace)) == (
        expected
    )


@pytest.mark.parametrize("seed", sorted(SPATIAL_BEST_OF_50))
def test_spatial_infomap_default_keeps_the_best_of_50_trials(seed):
    _, points = crowded(seed)
    expected = SPATIAL_BEST_OF_50[seed]
    for model in [
        SpatialInfomap(distance_metric="euclidean"),
        SpatialInfomap(distance_metric="euclidean", num_trials=50),
    ]:
        assert runs(point_labels(model.fit_predict(points))) == expected


@pytest.mark.parametrize("model", [Infostop, SpatialInfomap])
@pytest.mark.parametrize("value", [0, -1, 1.5, 50.0, "50", True, None])
def test_num_trials_must_be_a_positive_integer(model, value):
    with pytest.raises(AssertionError, match="num_trials"):
        model(num_trials=value)


@pytest.mark.parametrize("model", [Infostop, SpatialInfomap])
@pytest.mark.parametrize("value", [1, 7, np.int64(3)])
def test_num_trials_accepts_integers(model, value):
    model(num_trials=value)


@pytest.mark.parametrize("model", [Infostop, SpatialInfomap])
def test_num_trials_comes_last_so_positional_arguments_keep_their_meaning(model):
    params = list(inspect.signature(model).parameters)
    assert params[-2:] == ["verbose", "num_trials"]
