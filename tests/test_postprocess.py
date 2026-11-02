import warnings

import cpputils
import numpy as np
import pytest

from infostop import Infostop, SpatialInfomap, compute_intervals

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
    assert _intervals(
        [0, 0, -1, -1, 1, 1], [0, 1, 2, 3, 4, 5], max_time_between=10
    ) == [
        [0, 0, 1],
        [-1, 2, 3],
        [1, 4, 5],
    ]


def test_last_moving_interval_is_kept():
    assert _intervals([0, 0, -1, -1], [0, 1, 2, 3], max_time_between=10) == [
        [0, 0, 1],
        [-1, 2, 3],
    ]


@pytest.mark.parametrize("label", [0, -1])
@pytest.mark.parametrize(
    "times, expected_bounds",
    [
        ([0, 9], [[0, 9]]),
        ([0, 10], [[0, 10]]),
        ([0, 10.5], [[0, 0], [10.5, 10.5]]),
        ([0, 11], [[0, 0], [11, 11]]),
        # The gap is measured from the previous point, not from the start
        ([0, 10, 20], [[0, 20]]),
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_time_between": 10},
        {"model": Infostop(min_staying_time=5, max_time_between=10)},
    ],
    ids=["max_time_between", "model"],
)
def test_max_time_between_boundary(label, times, expected_bounds, kwargs):
    # Same rule as `cpputils.get_stationary_events`: a gap of exactly
    # `max_time_between` does not split an interval. Moving points (-1) follow the
    # same rule.
    expected = [[label, start, end] for start, end in expected_bounds]
    assert _intervals([label] * len(times), times, **kwargs) == expected


def test_empty_input():
    assert (
        compute_intervals(np.array([], dtype=int), np.array([]), max_time_between=10)
        == []
    )


@pytest.mark.parametrize("label", [-1, 3])
def test_single_point(label):
    assert _intervals([label], [5], max_time_between=10) == [[label, 5, 5]]


def _stop_events(trace, model, labels):
    """Return `[label, start_time, end_time]` of each stop event that `model` found."""
    _, event_map = cpputils.get_stationary_events(
        trace,
        model._r1,
        model._min_size,
        model._min_staying_time,
        model._max_time_between,
        "haversine",
    )
    event_map = np.array(event_map)
    return [
        [labels[event_map == j][0], *trace[event_map == j, 2][[0, -1]]]
        for j in range(event_map.max() + 1)
    ]


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
    events = _stop_events(trace, model, labels)
    assert len(events) == 2

    intervals = _intervals(labels, trace[:, 2], model=model)
    assert intervals == [events[0], [-1, move[0, 2], move[-1, 2]], events[1]]


def test_adjacent_stop_events_at_one_location_are_one_interval():
    # The position jumps ~15 m halfway through the stay at home. With `r1=10` that
    # starts a new stop event, but with `r2=30` both events get the home label.
    # `compute_intervals` only sees labels, so this is one stay at home.
    rng = np.random.default_rng(0)
    home_1 = _visit(rng, HOME, 0, n=15)
    home_2 = _visit(rng, (HOME[0] + 1.4e-4, HOME[1]), home_1[-1, 2] + 60, n=15)
    move = _move(HOME, WORK, home_2[-1, 2] + 60)
    work = _visit(rng, WORK, move[-1, 2] + 60)
    trace = np.vstack([home_1, home_2, move, work])

    model = Infostop(r1=10, r2=30)
    labels = model.fit_predict(trace)
    events = _stop_events(trace, model, labels)
    assert len(events) == 3 and events[0][0] == events[1][0] != events[2][0]

    assert _intervals(labels, trace[:, 2], model=model) == [
        [events[0][0], events[0][1], events[1][2]],
        [-1, move[0, 2], move[-1, 2]],
        events[2],
    ]


def test_model_with_millisecond_timestamps():
    # Timestamps in milliseconds, one point every 2 minutes. The 2-minute gaps are
    # longer than the default `max_time_between` (86400) but within the model's.
    rng = np.random.default_rng(0)
    home = _visit(rng, HOME, 0, n=15)
    move = _move(HOME, WORK, home[-1, 2] + 60)
    work = _visit(rng, WORK, move[-1, 2] + 60, n=15)
    trace = np.vstack([home, move, work])
    trace[:, 2] *= 2000

    model = Infostop(min_staying_time=300_000, max_time_between=86_400_000)
    labels = model.fit_predict(trace)
    intervals = _intervals(labels, trace[:, 2], model=model)
    assert intervals == _intervals(labels, trace[:, 2], max_time_between=86_400_000)
    assert [label for label, _, _ in intervals] == [labels[0], -1, labels[-1]]


def test_model_splits_at_a_gap_longer_than_its_max_time_between():
    # Two hours without data at home. With `max_time_between=3600` these are two
    # stays at home, also if both get the same label.
    rng = np.random.default_rng(0)
    home_1 = _visit(rng, HOME, 0, n=15)
    home_2 = _visit(rng, HOME, home_1[-1, 2] + 7200, n=15)
    trace = np.vstack([home_1, home_2])

    model = Infostop(max_time_between=3600)
    labels = model.fit_predict(trace)
    assert _intervals(labels, trace[:, 2], model=model) == [
        [labels[0], home_1[0, 2], home_1[-1, 2]],
        [labels[-1], home_2[0, 2], home_2[-1, 2]],
    ]


def test_neither_model_nor_max_time_between_warns_and_uses_86400():
    labels = np.array([0, 0, 0])
    times = np.array([0, 86400, 2 * 86400 + 1])
    with pytest.warns(FutureWarning, match="neither `model` nor `max_time_between`"):
        intervals = compute_intervals(labels, times)
    assert np.array(intervals).tolist() == [[0, 0, 86400], [0, 172801, 172801]]


def test_warning_points_at_the_caller():
    with pytest.warns(FutureWarning) as record:
        compute_intervals(np.array([0]), np.array([0]))
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "kwargs",
    [{"max_time_between": 86400}, {"model": Infostop()}],
    ids=["max_time_between", "model"],
)
def test_no_warning_when_given(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_intervals(np.array([0, 0]), np.array([0, 1]), **kwargs)


def test_model_and_max_time_between_raises():
    with pytest.raises(TypeError, match="not both"):
        compute_intervals(
            np.array([0]), np.array([0]), max_time_between=10, model=Infostop()
        )


def test_spatial_infomap_model_raises():
    with pytest.raises(TypeError, match="must be an `Infostop` model"):
        compute_intervals(np.array([0]), np.array([0]), model=SpatialInfomap())
