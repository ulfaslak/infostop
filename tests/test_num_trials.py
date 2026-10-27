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
import os
import subprocess
import sys

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


# Two people. Work is visited three times in all, home twice and the shop once, so
# work carries the most flow. Infomap 2.9 to 2.15 number the modules out of flow
# order when they run more than one trial; infostop renumbers them by flow.
HOME, WORK, SHOP = (0, 0), (400, 0), (0, 300)
ALICE = [(HOME, 10), (WORK, 10), (HOME, 10)]
BOB = [(WORK, 10), (SHOP, 10), (WORK, 10)]


@pytest.mark.parametrize("num_trials", [1, 2, 50])
def test_modules_are_numbered_by_flow_whatever_the_number_of_trials(num_trials):
    alice, bob = make_trace(1, ALICE), make_trace(2, BOB)
    labels_alice, labels_bob = Infostop(num_trials=num_trials).fit_predict([alice, bob])

    # Visits are 10 points; walks are -1. Work is module 1, home module 2, and the
    # shop, visited once, gets the next label after the modules.
    assert [l for l in labels_alice.tolist() if l != -1] == [2] * 10 + [1] * 10 + [
        2
    ] * 10
    assert [l for l in labels_bob.tolist() if l != -1] == [1] * 10 + [3] * 10 + [1] * 10


def test_bad_num_trials_is_rejected_under_python_dash_o():
    # Under `python -O` asserts are skipped; a bad value must still not reach
    # Infomap (Infomap 1.0.6 hangs on -1).
    code = (
        "from infostop import Infostop\n"
        "try:\n"
        "    Infostop(num_trials=-1)\n"
        "except AssertionError:\n"
        "    print('rejected')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    child = subprocess.run(
        [sys.executable, "-O", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.stdout.strip() == "rejected", (child.returncode, child.stderr)


def test_model_pickled_before_num_trials_still_fits():
    # Models pickled by older infostop versions have no `_num_trials`; they run
    # Infomap once, as they did when they were made.
    model = Infostop()
    del model._num_trials
    labels = model.fit_predict(make_trace(1, ALICE))
    assert sorted(set(labels.tolist())) == [-1, 1, 2]  # home, work (visited once)


@pytest.mark.parametrize("num_trials", [1, 50])
def test_singletons_keep_their_numbers(num_trials):
    # When no stop has a neighbour within r2, Infomap does not run and the labels
    # start at 0; renumbering the Infomap modules must not touch that path.
    trace = make_trace(1, [(HOME, 10), (WORK, 10)])
    labels = Infostop(num_trials=num_trials).fit_predict(trace)
    assert [l for l in labels.tolist() if l != -1] == [0] * 10 + [1] * 10
