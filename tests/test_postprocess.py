import cpputils
import numpy as np
import pytest

from infostop import Infostop, compute_intervals

HOME = (55.70, 12.50)
WORK = (55.71, 12.52)


def _visit(rng, location, t0, n=30):
    """`n` points scattered ~2 m around `location`, one per minute from `t0`."""
    return np.column_stack(
        [
            location[0] + rng.normal(0, 2e-5, n),
            location[1] + rng.normal(0, 2e-5, n),
            t0 + 60 * np.arange(n),
        ]
    )


def _move(a, b, t0, n=5):
    """`n` points ~280 m apart on the line from `a` to `b`, one per minute from `t0`."""
    f = np.linspace(0, 1, n + 2)[1:-1]
    return np.column_stack(
        [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), t0 + 60 * np.arange(n)]
    )


def _intervals(labels, times, **kwargs):
    return np.array(
        compute_intervals(np.array(labels), np.array(times), **kwargs)
    ).tolist()


def test_last_stop_interval_is_kept():
    assert _intervals([0, 0, -1, -1, 1, 1], [0, 1, 2, 3, 4, 5]) == [
        [0, 0, 1],
        [-1, 2, 3],
        [1, 4, 5],
    ]


def test_last_moving_interval_is_kept():
    assert _intervals([0, 0, -1, -1], [0, 1, 2, 3]) == [[0, 0, 1], [-1, 2, 3]]


@pytest.mark.parametrize(
    "gap, expected",
    [
        (9, [[0, 0, 9]]),
        (10, [[0, 0, 10]]),
        (11, [[0, 0, 0], [0, 11, 11]]),
    ],
)
def test_max_time_between_boundary(gap, expected):
    # Same rule as `cpputils.get_stationary_events`: a gap of exactly
    # `max_time_between` does not split a stop
    assert _intervals([0, 0], [0, gap], max_time_between=10) == expected


def test_empty_input():
    assert compute_intervals(np.array([], dtype=int), np.array([])) == []


@pytest.mark.parametrize("label", [-1, 3])
def test_single_point(label):
    assert _intervals([label], [5]) == [[label, 5, 5]]


def test_adjacent_stop_events_at_one_location_are_one_interval():
    # `compute_intervals` only sees labels: two stop events in a row that share a
    # label (e.g. split because a point was farther than `r1` from the median, but
    # within `r2`) are one stay at that location
    assert _intervals([2, 2, 2, 2], [0, 1, 2, 3]) == [[2, 0, 3]]


def test_stop_intervals_match_stop_events():
    # Home, with a gap of exactly `max_time_between` inside the visit, then work.
    # The trace ends in a stop.
    max_time_between = 3600
    rng = np.random.default_rng(0)
    home_1 = _visit(rng, HOME, 0, n=15)
    home_2 = _visit(rng, HOME, home_1[-1, 2] + max_time_between, n=15)
    move = _move(HOME, WORK, home_2[-1, 2] + 60)
    work = _visit(rng, WORK, move[-1, 2] + 60)
    trace = np.vstack([home_1, home_2, move, work])

    model = Infostop(max_time_between=max_time_between)
    labels = model.fit_predict(trace)
    _, event_map = cpputils.get_stationary_events(
        trace,
        model._r1,
        model._min_size,
        model._min_staying_time,
        max_time_between,
        "haversine",
    )
    event_map = np.array(event_map)
    events = [
        [labels[event_map == j][0], *trace[event_map == j, 2][[0, -1]]]
        for j in range(event_map.max() + 1)
    ]
    assert len(events) == 2

    intervals = _intervals(labels, trace[:, 2], max_time_between=max_time_between)
    assert [iv for iv in intervals if iv[0] != -1] == events
