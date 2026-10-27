import re

import numpy as np
import pytest

from infostop import Infostop

PLACES = [(55.70, 12.50), (55.71, 12.52), (55.69, 12.55)]


def make_trace(seed, sequence, n_per_visit):
    """Gaussian blobs of points at `PLACES[i]` for i in `sequence`, joined by moving
    segments (10 points along a straight line), one point per 60 s."""
    rng = np.random.default_rng(seed)
    parts, t = [], 0
    for i, (place, n) in enumerate(zip(sequence, n_per_visit)):
        if i > 0:
            a, b = np.array(PLACES[sequence[i - 1]]), np.array(PLACES[place])
            f = np.linspace(0, 1, 12)[1:-1, None]
            parts.append(np.column_stack([a + f * (b - a), t + 60 * np.arange(10)]))
            t += 600
        lat, lon = PLACES[place]
        parts.append(
            np.column_stack(
                [
                    lat + rng.normal(0, 2e-5, n),
                    lon + rng.normal(0, 2e-5, n),
                    t + 60 * np.arange(n),
                ]
            )
        )
        t += 60 * n + 60
    return np.vstack(parts)


# Users with different visits and lengths, so their label arrays differ
USERS = [
    make_trace(0, [0, 1, 0], [30, 20, 25]),
    make_trace(1, [1, 2, 1, 2], [15, 30, 10, 20]),
    make_trace(2, [2, 0], [40, 12]),
]


@pytest.mark.parametrize("n_users", [1, 2, 3])
def test_tuple_gives_same_labels_as_list(n_users):
    traces = USERS[:n_users]
    labels_list = Infostop().fit_predict(list(traces))
    labels_tuple = Infostop().fit_predict(tuple(traces))
    assert type(labels_tuple) is tuple
    assert len(labels_tuple) == n_users
    for a, b, trace in zip(labels_list, labels_tuple, traces):
        assert isinstance(b, np.ndarray)
        assert len(b) == len(trace)
        np.testing.assert_array_equal(a, b)
        assert -1 in b and len(set(b) - {-1}) > 0  # moving and stationary points


@pytest.mark.parametrize("container", [list, tuple])
def test_return_type_matches_input_type(container):
    traces = container(USERS)
    assert type(Infostop().fit_predict(traces)) == type(traces)
    assert type(Infostop().fit_predict(USERS[0])) == type(USERS[0])


def shape(a):
    return re.escape(str(np.shape(a)))


@pytest.mark.parametrize(
    "data, got",
    [
        # A single trace as a plain list of [lat, lon, t] rows is read as one user
        # per row. It must fail clearly, not be reinterpreted as one trace.
        (USERS[0].tolist(), r"User 0: .*got a list\."),
        (tuple(map(tuple, USERS[0])), r"User 0: .*got a tuple\."),
        # Several users, each a list of rows
        ([USERS[0].tolist(), USERS[0].tolist()], r"User 0: .*got a list\."),
        # Columns passed separately
        (
            tuple(USERS[0].T),
            rf"User 0: .*got an array of shape {shape(USERS[0][:, 0])}\.",
        ),
        # A single row
        (USERS[0][0], rf"^Each .*got an array of shape {shape(USERS[0][0])}\."),
        # Users of the same length stacked into one 3D array
        (np.stack([USERS[0], USERS[0]]), rf"shape {shape([USERS[0], USERS[0]])}\."),
    ],
)
def test_input_that_is_not_a_2d_array_fails_clearly(data, got):
    with pytest.raises(AssertionError, match=got) as excinfo:
        Infostop().fit_predict(data)
    assert "must be a 2D numpy array with one row per point" in str(excinfo.value)
