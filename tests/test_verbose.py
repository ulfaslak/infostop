import re

import numpy as np
import pytest

from infostop import Infostop, utils
from infostop.models import SpatialInfomap

HOME, WORK, SHOP = (55.70, 12.50), (55.71, 12.52), (55.72, 12.50)


def make_trace(places, seed=0, n=30):
    """One Gaussian blob of `n` points per visit, 60 s apart, visits 4000 s apart."""
    rng = np.random.default_rng(seed)
    visits = []
    for i, (lat, lon) in enumerate(places):
        visits.append(
            np.column_stack(
                [
                    lat + rng.normal(0, 2e-5, n),
                    lon + rng.normal(0, 2e-5, n),
                    4000 * i + 60 * np.arange(n),
                ]
            )
        )
    return np.vstack(visits)


def printed_count(capsys):
    """The N in the one line `Found N stop locations`, which must start a line."""
    out = capsys.readouterr().out
    assert out.count("Found") == 1
    counts = re.findall(r"^Found (\d+) stop locations$", out, re.M)
    assert len(counts) == 1
    return int(counts[0])


def stop_locations(labels):
    """The number of distinct labels other than -1, over all users."""
    if isinstance(labels, (list, tuple)):
        labels = np.concatenate(labels)
    return len(set(labels.tolist()) - {-1})


@pytest.mark.parametrize(
    "places, label_singleton, expected",
    [
        ([HOME, WORK, HOME], True, 2),  # home revisited, work visited once
        ([HOME, WORK, HOME], False, 1),  # work, visited once, is -1
        ([HOME, WORK, HOME, WORK], True, 2),  # both revisited
        ([HOME, WORK, HOME, WORK], False, 2),
        ([HOME, WORK, SHOP, HOME], True, 3),  # work and shop visited once
        ([HOME, WORK, SHOP, HOME], False, 1),
        ([HOME, WORK], True, 2),  # nothing revisited: Infomap does not run
        ([HOME, WORK], False, 0),
    ],
)
def test_infostop_verbose_count(capsys, places, label_singleton, expected):
    labels = Infostop(verbose=True, label_singleton=label_singleton).fit_predict(
        make_trace(places)
    )
    assert stop_locations(labels) == expected
    assert printed_count(capsys) == expected


def test_infostop_verbose_count_revisits_at_one_coordinate(capsys):
    # With `label_singleton=False`, a place visited twice is a stop location even
    # when its visits round to one coordinate, a node with no neighbor (#68)
    model = Infostop(verbose=True, label_singleton=False, min_spacial_resolution=1e-3)
    labels = model.fit_predict(make_trace([HOME, WORK, HOME]))
    assert stop_locations(labels) == 1
    assert printed_count(capsys) == 1


@pytest.mark.parametrize("label_singleton", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_infostop_verbose_count_matches_labels(capsys, seed, label_singleton):
    rng = np.random.default_rng(seed)
    places = [(55.70 + 0.01 * i, 12.50) for i in rng.integers(0, 6, size=8)]
    model = Infostop(verbose=True, label_singleton=label_singleton)
    labels = model.fit_predict(make_trace(places, seed=seed))
    assert printed_count(capsys) == stop_locations(labels)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("label_singleton", [True, False])
def test_infostop_verbose_count_multiuser(capsys, label_singleton, container):
    # Users share labels, so a place that two users visit once each is one stop
    # location visited twice; a place one user visits once is a singleton
    traces = container(
        [
            make_trace([HOME, WORK], seed=1),
            make_trace([WORK, SHOP], seed=2),
            make_trace([(55.73, 12.50)], seed=3),
        ]
    )
    model = Infostop(verbose=True, label_singleton=label_singleton)
    labels = model.fit_predict(traces)
    assert stop_locations(labels) == (4 if label_singleton else 1)
    assert printed_count(capsys) == stop_locations(labels)


@pytest.mark.parametrize("label_singleton", [True, False])
def test_spatial_infomap_verbose_count(capsys, label_singleton):
    # Two blobs and one point on its own
    rng = np.random.default_rng(0)
    points = np.vstack(
        [
            rng.normal(0, 0.5, (6, 2)),
            rng.normal([50, 50], 0.5, (3, 2)),
            [[100, 0]],
        ]
    )
    model = SpatialInfomap(
        r2=5, distance_metric="euclidean", label_singleton=label_singleton, verbose=True
    )
    labels = model.fit_predict(points)
    assert stop_locations(labels) == (3 if label_singleton else 2)
    assert printed_count(capsys) == stop_locations(labels)


@pytest.mark.parametrize("label_singleton", [True, False])
def test_spatial_infomap_verbose_count_all_singletons(capsys, label_singleton):
    # No point within `r2` of another: the network is empty and Infomap does not run
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    model = SpatialInfomap(
        r2=1, distance_metric="euclidean", label_singleton=label_singleton, verbose=True
    )
    labels = model.fit_predict(points)
    assert stop_locations(labels) == (4 if label_singleton else 0)
    assert printed_count(capsys) == stop_locations(labels)


@pytest.mark.parametrize("places", [[HOME, WORK, HOME], [HOME, WORK]])
def test_running_infomap_line_ends(capsys, places):
    # The line ends whether or not Infomap runs, so the count starts its own line
    Infostop(verbose=True).fit_predict(make_trace(places))
    lines = capsys.readouterr().out.splitlines()
    running = [line for line in lines if "running Infomap" in line]
    assert len(running) == 1
    if len(places) > len(set(places)) and utils._INFOMAP_LOGS_TO_PYTHON:
        # Infomap's log comes through Python here, so it starts on a new line
        assert running[0] == "    ... running Infomap..."
        assert lines.index("done") > lines.index(running[0])
    elif len(places) > len(set(places)):
        # Infomap 2.14 and older print their log from C++, outside `sys.stdout`
        assert running[0] == "    ... running Infomap... done"
    else:
        assert running[0] == "    ... running Infomap... skipped (no nodes)"
