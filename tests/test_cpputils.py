"""Edge cases of the C++ step, `cpputils.get_stationary_events`.

`get_stationary_events(coords, r_C, min_size, min_staying_time, max_staying_time,
distance_metric)` walks the points in order. A point joins the current group if it
is within `r_C` (`<=`) of the group's coordinate-wise median and, with a time
column, at most `max_staying_time` (`<=`) after the previous point. A finished
group is a stop event if it has at least `min_size` points and, with a time
column, lasts at least `min_staying_time` from first to last point.

It returns `(stat_coords, event_map)`: the median of each stop event, and for each
point the index of its stop event or -1.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.lib.stride_tricks import as_strided

import cpputils

LAT0, LON0 = 55.6761, 12.5683
M_PER_DEG_LAT = 6371000 * np.pi / 180


def events(
    coords,
    r1=10,
    min_size=2,
    min_staying_time=300,
    max_time_between=86400,
    metric="euclidean",
):
    return cpputils.get_stationary_events(
        np.asarray(coords, dtype=float),
        r1,
        min_size,
        min_staying_time,
        max_time_between,
        metric,
    )


@pytest.mark.parametrize("n_cols", [2, 3])
def test_zero_points(n_cols):
    assert events(np.empty((0, n_cols))) == ([], [])


@pytest.mark.parametrize("n_cols", [2, 3])
def test_zero_points_invents_no_event(n_cols):
    # A zero-row view into the middle of a larger array, so an out-of-bounds read
    # would hit valid memory and show up as a stop event made from rows that are not
    # in the input: row 0 starts the first group, and with 3 columns the last-group
    # check reads row -1 (time 1000). `min_size=0` lets such a group through. This
    # only catches reads that change the output (#48).
    parent = np.array([[9.0, 9.0, 1000.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    empty = as_strided(parent[1:, :n_cols], shape=(0, n_cols))
    assert events(empty, min_size=0) == ([], [])


def test_one_point():
    assert events([[1, 2]]) == ([], [-1])
    assert events([[1, 2, 0]]) == ([], [-1])
    assert events([[1, 2]], min_size=1) == ([[1.0, 2.0]], [0])
    # With a time column a single point lasts 0 seconds.
    assert events([[1, 2, 0]], min_size=1) == ([], [-1])
    assert events([[1, 2, 0]], min_size=1, min_staying_time=0) == ([[1.0, 2.0]], [0])


@pytest.mark.parametrize("n_cols", [2, 3])
def test_one_event_covers_everything(n_cols):
    xy = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 1]]
    coords = [p + [600 * i] for i, p in enumerate(xy)] if n_cols == 3 else xy
    # The median is taken per coordinate: x of [0, 0, 1, 1, 2], y of [0, 0, 1, 1, 1].
    assert events(coords) == ([[1.0, 1.0]], [0, 0, 0, 0, 0])


def test_median_of_even_count_averages_the_middle_pair():
    assert events([[0, 0], [1, 4], [3, 5], [2, 9]]) == ([[1.5, 4.5]], [0, 0, 0, 0])


def test_last_group():
    # The trace ends inside a stop: the last group is a stop event.
    assert events([[0, 0], [0, 1], [100, 0], [100, 1]]) == (
        [[0.0, 0.5], [100.0, 0.5]],
        [0, 0, 1, 1],
    )
    # The trace ends with a group that is too small: it is -1.
    assert events([[0, 0], [0, 1], [100, 0]]) == ([[0.0, 0.5]], [0, 0, -1])


@pytest.mark.parametrize("min_size", [2, 3, 5])
def test_min_size_boundary(min_size):
    big = [[0, i] for i in range(min_size)]
    small = [[100, i] for i in range(min_size - 1)]
    big_median = [0.0, (min_size - 1) / 2]

    # A group of exactly `min_size` points is a stop event, one fewer is not,
    # both in the middle of the trace and as the last group.
    assert events(big + small, min_size=min_size) == (
        [big_median],
        [0] * min_size + [-1] * (min_size - 1),
    )
    assert events(small + big, min_size=min_size) == (
        [big_median],
        [-1] * (min_size - 1) + [0] * min_size,
    )


def test_distance_exactly_r1_joins():
    assert events([[0, 0], [0, 10]], r1=10) == ([[0.0, 5.0]], [0, 0])
    assert events([[0, 0], [0, 10]], r1=np.nextafter(10, 0)) == ([], [-1, -1])


def test_distance_is_measured_to_the_group_median():
    # The third point is within r1 of the previous point (8) but not of the first
    # point (16 and 13), and what counts is the median of the group so far (y = 4).
    assert events([[0, 0], [0, 8], [0, 13]]) == ([[0.0, 8.0]], [0, 0, 0])
    assert events([[0, 0], [0, 8], [0, 16]]) == ([[0.0, 4.0]], [0, 0, -1])
    # The last point is 9 from the median (y = 1) and 11 from the mean (y = 3).
    assert events([[0, 0], [0, 1], [0, 8], [0, -8]]) == ([[0.0, 0.5]], [0, 0, 0, 0])


@pytest.mark.parametrize("metres, joined", [(9.9, True), (10.1, False)])
def test_haversine_r1_in_metres(metres, joined):
    coords = [[LAT0, LON0], [LAT0 + metres / M_PER_DEG_LAT, LON0]]
    stat_coords, event_map = events(coords, metric="haversine")
    assert event_map == ([0, 0] if joined else [-1, -1])
    assert len(stat_coords) == int(joined)


@pytest.mark.parametrize("gap, joined", [(600, True), (601, False)])
def test_time_gap_around_max_time_between(gap, joined):
    coords = [[0, 0, 0], [0, 0, 300], [0, 0, 300 + gap]]
    expected_map = [0, 0, 0] if joined else [0, 0, -1]
    assert events(coords, max_time_between=600) == ([[0.0, 0.0]], expected_map)


def test_time_gap_is_measured_between_consecutive_points():
    # Each gap is 600 (allowed) although the group spans 1200 > max_time_between.
    coords = [[0, 0, 0], [0, 0, 600], [0, 0, 1200]]
    assert events(coords, max_time_between=600) == ([[0.0, 0.0]], [0, 0, 0])


@pytest.mark.parametrize("first, last", [(300, 299), (299, 300)])
def test_min_staying_time_boundary(first, last):
    # Two groups, 100 apart. A group that lasts exactly `min_staying_time` is a stop
    # event, one second less is not, both in the middle and as the last group.
    coords = [[0, 0, 0], [0, 0, first], [100, 0, 1000], [100, 0, 1000 + last]]
    stat_coords, event_map = events(coords, min_staying_time=300)
    if first == 300:
        assert (stat_coords, event_map) == ([[0.0, 0.0]], [0, 0, -1, -1])
    else:
        assert (stat_coords, event_map) == ([[100.0, 0.0]], [-1, -1, 0, 0])


def test_time_column_does_not_change_grouping_when_within_limits():
    xy = [[0, 0], [0, 1], [50, 0], [100, 0], [100, 2], [100, 1]]
    with_time = [p + [600 * i] for i, p in enumerate(xy)]
    expected = ([[0.0, 0.5], [100.0, 1.0]], [0, 0, -1, 1, 1, 1])
    assert events(xy) == expected
    assert events(with_time) == expected


def test_integer_and_list_input_match_float_array():
    xy = [[0, 0], [0, 1], [50, 0], [100, 0], [100, 2]]
    expected = events(np.array(xy, dtype=float))
    assert events(np.array(xy, dtype=int)) == expected
    assert (
        cpputils.get_stationary_events(xy, 10, 2, 300, 86400, "euclidean") == expected
    )


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="column counts other than 2 and 3 are not rejected; "
    "https://github.com/ulfaslak/infostop/issues/49",
)
@pytest.mark.parametrize("n_cols", [1, 4])
def test_wrong_column_count_raises(n_cols):
    with pytest.raises(ValueError):
        events(np.zeros((3, n_cols)))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="an unknown distance_metric calls an uninitialised function pointer; "
    "https://github.com/ulfaslak/infostop/issues/49",
)
def test_unknown_metric_raises():
    # In a subprocess: the call is undefined behaviour and may crash the process.
    # The child exits 0 if it raises ValueError and 3 if nothing is raised.
    code = (
        "import sys, numpy as np, cpputils\n"
        "try:\n"
        "    cpputils.get_stationary_events(np.zeros((3, 2)), 10, 2, 300, 86400, 'Euclidean')\n"
        "except ValueError:\n"
        "    sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    child = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # A crash is the bug too: a negative code is a POSIX signal, and a code of
    # 0xC0000000 or more is a Windows NTSTATUS (e.g. 0xC0000005, access violation).
    crashed = child.returncode < 0 or child.returncode >= 0xC0000000
    if child.returncode not in (0, 3) and not crashed:
        # A Python error in the child (e.g. cpputils not importable), not the bug.
        raise RuntimeError(f"exit code {child.returncode}: {child.stderr}")
    assert child.returncode == 0, f"exit code {child.returncode}"
